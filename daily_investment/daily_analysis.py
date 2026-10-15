#!/usr/bin/env python3
"""Read-only market monitor. Python 3.10+, standard library only.

Public quote endpoints are unofficial and may change. Missing data fails closed.
No orders, no fabricated news, no premium inferred from an old daily NAV.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

CN = timezone(timedelta(hours=8))
ROOT = Path(__file__).resolve().parent


def stamp(value):
    value = str(value).strip()
    if re.fullmatch(r"\d{14}", value):
        return datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=CN)
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("时间必须包含时区，例如 2026-10-13T09:50:00+08:00")
    return dt.astimezone(CN)


def number(value):
    n = float(value)
    if not math.isfinite(n):
        raise ValueError("数值必须有限")
    return n


def validate_config(config):
    if not config.get("assets"):
        raise ValueError("assets 不可为空")
    ids = [a["id"] for a in config["assets"]]
    if len(set(ids)) != len(ids):
        raise ValueError("asset id 不能重复")
    if config.get("timezone") != "Asia/Shanghai":
        raise ValueError("本工具仅支持北京时间")
    weights = []
    for key in ("quote_max_age_minutes", "reference_max_gap_minutes", "premium_alert_pct"):
        config[key] = number(config[key])
        if config[key] <= 0:
            raise ValueError(key + " 必须为正数")
    config["portfolio_growth_alert_weight"] = number(config["portfolio_growth_alert_weight"])
    if not 0 < config["portfolio_growth_alert_weight"] <= 1:
        raise ValueError("portfolio_growth_alert_weight 必须在0至1之间")
    for a in config["assets"]:
        if not re.fullmatch(r"[a-z0-9_]+", a["id"]):
            raise ValueError("asset id 格式错误")
        if not re.fullmatch(r"(sh|sz|bj|hk)[A-Za-z0-9]+", a["symbol"]):
            raise ValueError("行情代码格式错误")
        if a.get("weight") is not None:
            w = number(a["weight"])
            if not 0 <= w <= 1:
                raise ValueError("weight 用小数，20% 填 0.20")
            if a.get("proxy"):
                raise ValueError("代理标的不能填写真实持仓权重，请先替换为实际基金并取消 proxy")
            weights.append(w)
            a["weight"] = w
        if a.get("cost") is not None and number(a["cost"]) <= 0:
            raise ValueError("cost 必须为正数")
        if a.get("cost") is not None:
            a["cost"] = number(a["cost"])
        if a.get("is_index") and a.get("cost") is not None:
            raise ValueError("指数点位不能与基金成本比较；请先替换为实际基金并取消 is_index")
    if sum(weights) > 1.000001:
        raise ValueError("持仓权重合计不能超过100%")


def fetch(url, timeout):
    request = Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.qq.com/"})
    for attempt in range(2):
        try:
            with urlopen(request, timeout=timeout) as r:
                return r.read()
        except Exception:
            if attempt:
                raise
            time.sleep(0.3)


def parse_tencent(payload, asset):
    match = re.search(r'="([^"]*)"', payload)
    fields = match.group(1).split("~") if match else []
    if len(fields) < 31:
        raise ValueError("行情结构不支持或代码不可用")
    if fields[2] != asset["symbol"][2:]:
        raise ValueError("返回代码与请求不一致")
    price, prev = number(fields[3]), number(fields[4])
    if price <= 0 or prev <= 0:
        raise ValueError("无有效成交价或昨收价")
    provider_time = fields[30]
    if asset["symbol"].startswith("hk") and re.fullmatch(r"\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}", provider_time):
        # Tencent Hong Kong quotes use local UTC+8 time with slash separators.
        provider_time = datetime.strptime(provider_time, "%Y/%m/%d %H:%M:%S").replace(tzinfo=CN).isoformat()
    return {"price": price, "previous_close": prev,
            "asof": stamp(provider_time).isoformat(), "currency": asset["currency"],
            "status": "ok", "provider_name": fields[1]}


def online_quote(asset, timeout):
    url = "https://qt.gtimg.cn/q=" + asset["symbol"]
    quote = parse_tencent(fetch(url, timeout).decode("gb18030"), asset)
    quote["source_url"] = url
    return quote


def online_history(asset, timeout):
    if not asset.get("secid"):
        if asset["symbol"].startswith("hk"):
            url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?" + urlencode({"param": asset["symbol"] + ",day,,,260,qfq"})
            payload = json.loads(fetch(url, timeout))
            data = payload.get("data", {}).get(asset["symbol"], {})
            bars = data.get("qfqday") or data.get("day")
            if payload.get("code") != 0 or not bars:
                raise ValueError("香港日线为空或返回标的不匹配")
            adjustment = "qfq" if data.get("qfqday") else "price"
            rows = [{"date": date.fromisoformat(f[0]).isoformat(), "close": number(f[2]),
                     "adjustment": adjustment, "source_url": url} for f in bars]
            return sorted(rows, key=lambda r: r["date"])[-260:]
        return []
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get?" + urlencode({
        "secid": asset["secid"], "klt": "101", "fqt": "1", "lmt": "260",
        "beg": "19900101", "end": "20500101", "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"})
    data = json.loads(fetch(url, timeout)).get("data")
    if not data or not data.get("klines"):
        raise ValueError("日线为空")
    if str(data.get("code")) != asset["symbol"][2:]:
        raise ValueError("日线返回代码不一致")
    rows = []
    for line in data["klines"]:
        f = line.split(",")
        rows.append({"date": date.fromisoformat(f[0]).isoformat(),
                     "close": number(f[2]), "adjustment": "qfq", "source_url": url})
    # Some responses ignore lmt when beg/end are supplied. Bound the sample ourselves.
    return sorted(rows, key=lambda r: r["date"])[-260:]


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def local_inputs(folder, known_ids):
    quotes, histories = {}, {}
    for row in read_csv(folder / "quotes.csv"):
        key = row["asset_id"]
        if key not in known_ids or key in quotes:
            raise ValueError("报价 asset_id 未知或重复")
        if row["status"] not in {"ok", "suspended"}:
            raise ValueError("status 只能为 ok 或 suspended")
        price, prev = number(row["price"]), number(row["previous_close"])
        if min(price, prev) <= 0:
            raise ValueError("报价必须为正数")
        quotes[key] = {**row, "price": price, "previous_close": prev,
                       "asof": stamp(row["asof"]).isoformat()}
    for row in read_csv(folder / "history.csv"):
        key = row["asset_id"]
        if key not in known_ids:
            raise ValueError("日线 asset_id 未知")
        if row["adjustment"] not in {"qfq", "price", "total_return"}:
            raise ValueError("adjustment 只能为 qfq/price/total_return")
        close = number(row["close"])
        if close <= 0:
            raise ValueError("日线必须为正数")
        histories.setdefault(key, []).append({**row, "close": close,
                                               "date": date.fromisoformat(row["date"]).isoformat()})
    return quotes, histories


def metrics(rows, now):
    # Always use completed prior-day bars, even when the endpoint includes today's partial bar.
    rows = sorted((r for r in rows if date.fromisoformat(r["date"]) < now.date()), key=lambda r: r["date"])
    dates = [r["date"] for r in rows]
    if len(dates) != len(set(dates)):
        raise ValueError("日线日期重复")
    if len({r["adjustment"] for r in rows}) > 1:
        raise ValueError("日线复权口径混杂")
    if not rows:
        return {"status": "missing"}
    prices = [number(r["close"]) for r in rows]
    if min(prices) <= 0:
        raise ValueError("日线价格无效")
    result = {"status": "ok", "asof": dates[-1], "observations": len(prices),
              "adjustment": rows[-1]["adjustment"], "last": prices[-1]}
    for n in (5, 20, 60):
        result[f"return_{n}d_pct"] = (prices[-1] / prices[-n-1] - 1) * 100 if len(prices) > n else None
        result[f"ma{n}"] = statistics.mean(prices[-n:]) if len(prices) >= n else None
    peak, drawdown = prices[0], 0.0
    for p in prices:
        peak = max(peak, p)
        drawdown = min(drawdown, p / peak - 1)
    result["max_drawdown_pct"] = drawdown * 100
    returns = [math.log(b / a) for a, b in zip(prices, prices[1:])]
    result["annualized_vol_pct"] = statistics.stdev(returns) * math.sqrt(252) * 100 if len(returns) >= 20 else None
    ma20, ma60 = result["ma20"], result["ma60"]
    result["trend"] = ("偏强" if prices[-1] > ma20 > ma60 else
                       "偏弱" if prices[-1] < ma20 < ma60 else "震荡") if ma20 and ma60 else "样本不足"
    result["source_url"] = rows[-1].get("source_url", "")
    return result


def premium(quote, reference, now, config):
    if not quote or not reference:
        return {"status": "missing", "reason": "未提供有效同期参考价值"}
    if reference.get("kind") not in {"iopv", "fair_value"}:
        return {"status": "invalid", "reason": "旧日净值不能直接用来计算实时溢价"}
    if reference.get("currency") != quote.get("currency"):
        return {"status": "invalid", "reason": "参考价值与价格币种不同"}
    if not reference.get("source_url"):
        return {"status": "invalid", "reason": "参考价值缺少来源"}
    try:
        qt, rt = stamp(quote["asof"]), stamp(reference["asof"])
        ut = stamp(reference["underlying_asof"])
        value = number(reference["reference_value"])
        if value <= 0:
            raise ValueError("参考价值必须为正数")
        if max(qt, rt, ut) > now + timedelta(minutes=2):
            raise ValueError("参考时间在未来")
        gap = config["reference_max_gap_minutes"]
        if abs((qt - rt).total_seconds()) > gap * 60 or abs((rt - ut).total_seconds()) > gap * 60:
            raise ValueError("报价、参考价值或底层估值时间不同步")
        if now - qt > timedelta(minutes=config["quote_max_age_minutes"]):
            raise ValueError("报价已过期")
        if quote.get("status") != "ok":
            raise ValueError("停牌报价不能用于实时溢价")
        return {"status": "ok", "pct": (quote["price"] / value - 1) * 100,
                "source_url": reference["source_url"], "asof": rt.isoformat(), "kind": reference["kind"]}
    except (ValueError, KeyError) as e:
        return {"status": "invalid", "reason": str(e)}


def analyze(asset, quote, rows, reference, now, config, errors):
    quality = "缺失"
    if quote:
        qt = stamp(quote["asof"])
        if quote.get("currency") != asset["currency"]:
            quality = "币种不一致"
        elif qt > now + timedelta(minutes=2):
            quality = "时间异常"
        elif quote.get("status") == "suspended":
            quality = "停牌/旧成交"
        elif qt.date() != now.date():
            quality = "历史报价"
        elif now - qt > timedelta(minutes=config["quote_max_age_minutes"]):
            quality = "延迟/需核实"
        else:
            quality = "当日有效"
    try:
        trend = metrics(rows, now)
    except ValueError as e:
        errors.append("日线校验失败: " + str(e))
        trend = {"status": "invalid"}
    prem = premium(quote if quality == "当日有效" else None, reference, now, config)
    notes = []
    if asset.get("proxy"):
        notes.append("代理资产，不能视为本人实际基金；油气股票不等于原油现货")
    if quality != "当日有效":
        notes.append("暂停依据当前价格形成操作提示；先核实数据或停复牌状态")
    fresh_history = bool(trend.get("asof")) and (now.date() - date.fromisoformat(trend["asof"])).days <= 7
    if trend.get("asof") and not fresh_history:
        notes.append("日线超过7个自然日，趋势仅供历史参考")
    if prem["status"] == "ok" and prem["pct"] > config["premium_alert_pct"]:
        notes.append("溢价超过自定义警戒线：暂停追买，复核后评估减仓/替代产品")
    elif asset["id"] == "nasdaq_tech" and prem["status"] != "ok":
        notes.append("实时溢价未核实：不输出估算溢价，先查基金管理人公告及可靠IOPV")
    if quality == "当日有效" and fresh_history and trend.get("trend") == "偏弱":
        notes.append("历史日线偏弱：暂缓集中补仓；若超配，结合业绩和反弹承接评估再平衡")
    elif quality == "当日有效" and fresh_history and trend.get("trend") == "偏强":
        notes.append("历史日线偏强：可保留合理底仓，避免仅依据趋势追涨")
    if asset["group"] == "long_bond":
        notes.append("与另一国债ETF合并管理久期；30年品种利率敏感度更高")
    if asset.get("is_index"):
        notes.append("指数点位仅用于市场观察，实际基金可能存在跟踪误差")
    if not notes:
        notes.append("观察，需结合新闻、盈利、估值和真实持仓复核")
    pnl = None
    if quality == "当日有效" and quote and asset.get("cost") and not asset.get("proxy"):
        pnl = (quote["price"] / asset["cost"] - 1) * 100
    return {"asset": asset, "quote": quote, "quote_quality": quality,
            "metrics": trend, "premium": prem, "price_pnl_pct": pnl,
            "notes": notes, "errors": errors}


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def fmt(value, suffix=""):
    return "未核实" if value is None else f"{value:.2f}{suffix}"


def markdown(results, now, config):
    lines = [f"# 每日投资数据检查 {now:%Y-%m-%d %H:%M} 北京时间", "",
             "本报告为规则检查，不是自动交易指令。9:50属于盘中，跨境ETF可能尚未复牌。",
             "当日有效仅表示报价通过时间与币种检查；公开报价接口不独立确认停复牌，请核实交易所/管理人公告。",
             "每个报价分别列明时间。日线指标仅使用今天之前的完整日线；价格涨幅不等于基金总收益。", "",
             "|资产|价格/指数|当日涨跌|报价时间|质量|历史趋势|20交易日变化|提示|",
             "|---|---:|---:|---|---|---|---:|---|"]
    for r in results:
        q, a, m = r["quote"], r["asset"], r["metrics"]
        change = (q["price"] / q["previous_close"] - 1) * 100 if q and r["quote_quality"] == "当日有效" else None
        lines.append("|" + "|".join(map(cell, [a["name"], fmt(q["price"] if q else None),
                     fmt(change, "%"), q["asof"] if q else "—", r["quote_quality"],
                     m.get("trend", "未核实"), fmt(m.get("return_20d_pct"), "%"), "；".join(r["notes"])])) + "|")
    lines += ["", "## 数据与溢价核对", ""]
    for r in results:
        a, q, m, p = r["asset"], r["quote"], r["metrics"], r["premium"]
        lines.append(f"- {a['name']}：日线截止 {m.get('asof', '缺失')}，{m.get('observations', 0)} 条；口径 {m.get('adjustment', '未知')}。")
        lines.append(f"  MA5/20/60：{fmt(m.get('ma5'))} / {fmt(m.get('ma20'))} / {fmt(m.get('ma60'))}；5/20/60交易日变化：{fmt(m.get('return_5d_pct'), '%')} / {fmt(m.get('return_20d_pct'), '%')} / {fmt(m.get('return_60d_pct'), '%')}；样本最大回撤：{fmt(m.get('max_drawdown_pct'), '%')}；样本年化波动：{fmt(m.get('annualized_vol_pct'), '%')}。")
        if q:
            lines.append(f"  报价来源：{q.get('source_url', '未提供')}；盈亏粗估：{fmt(r['price_pnl_pct'], '%')}（未计分红、手续费）。")
        if p["status"] == "ok":
            lines.append(f"  同期参考溢价：{p['pct']:.2f}%；来源：{p['source_url']}。跨境IOPV仍需确认是否覆盖海外最新变动。")
        else:
            lines.append(f"  溢价：未核实；{p['reason']}。")
        if m.get("source_url"):
            lines.append("  日线来源：" + m["source_url"])
        lines.extend("  数据异常：" + cell(e) for e in r["errors"])
    assets = config["assets"]
    growth = sum(a.get("weight") or 0 for a in assets if a["group"] == "growth")
    long_bond = sum(a.get("weight") or 0 for a in assets if a["group"] == "long_bond")
    missing = [a["name"] for a in assets if a.get("weight") is None and not a.get("proxy")]
    lines += ["", "## 持仓检查", "",
              f"已填写的成长类权重合计 {growth:.1%}，长债权重合计 {long_bond:.1%}。",
              "权重未全部填写时，以上是已知部分，不能代表完整账户风险。",
              "红利与油气可能有行业重叠；纳指科技、恒科、创业板、北证50可能同步下跌。"]
    if missing:
        lines.append("未填写仓位：" + "、".join(missing))
    if growth > config["portfolio_growth_alert_weight"]:
        lines.append("成长类超过自定义检查线：复核集中度；该阈值不是自动卖出比例。")
    ranked = sorted((r for r in results if r["metrics"].get("return_20d_pct") is not None
                     and (now.date() - date.fromisoformat(r["metrics"]["asof"])).days <= 7),
                    key=lambda r: r["metrics"]["return_20d_pct"], reverse=True)
    lines += ["", "## 关注资产相对强弱", "", "仅按已配置资产的20交易日价格变化排序，不代表全市场热门板块，也不能跨资产推断估值高低。"]
    lines += [f"- {r['asset']['name']}：{r['metrics']['return_20d_pct']:.2f}%" for r in ranked] or ["无足够的新鲜日线样本。"]
    lines += ["", "## 每日人工/AI补充", "",
              "请核实基金溢价和停复牌公告、国内10/30年收益率、海外昨收与美元、黄金和原油、三季报/政策及板块强弱。",
              "新闻与买卖判断不由此脚本凭空生成；不要根据均线或单日涨跌单独下单。",
              "所有代理标的必须替换为实际基金后，才能做本人持仓分析。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--input-dir", type=Path, help="用自己的CSV离线分析，不联网")
    parser.add_argument("--references", type=Path, help="同期IOPV/公允价值CSV")
    parser.add_argument("--out", type=Path, default=ROOT / "reports")
    parser.add_argument("--now", help="指定带时区的时点，仅用于重放/测试")
    parser.add_argument("--timeout", type=float, default=8)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout 必须为正数")
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    validate_config(config)
    now = stamp(args.now) if args.now else datetime.now(CN)
    known = {a["id"] for a in config["assets"]}
    quotes, histories = local_inputs(args.input_dir, known) if args.input_dir else ({}, {})
    references = {}
    for ref in read_csv(args.references) if args.references else []:
        if ref["asset_id"] not in known or ref["asset_id"] in references:
            raise ValueError("参考值 asset_id 未知或重复")
        references[ref["asset_id"]] = ref
    def collect(a):
        errors, q, rows = [], quotes.get(a["id"]), histories.get(a["id"], [])
        if not args.input_dir:
            try:
                q = online_quote(a, args.timeout)
            except Exception as e:
                errors.append("报价采集失败: " + str(e))
            try:
                rows = online_history(a, args.timeout)
            except Exception as e:
                errors.append("日线采集失败: " + str(e))
        return analyze(a, q, rows, references.get(a["id"]), now, config, errors)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(collect, config["assets"]))
    args.out.mkdir(parents=True, exist_ok=True)
    stem = now.strftime("%Y%m%d_%H%M%S_%f")
    report = args.out / (stem + ".md")
    report.write_text(markdown(results, now, config), encoding="utf-8")
    data = {"generated_at": now.isoformat(), "results": results,
            "valid_quote_count": sum(r["quote_quality"] == "当日有效" for r in results)}
    (args.out / (stem + ".json")).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告: {report}\n有效当日报价: {data['valid_quote_count']}/{len(results)}")
    # A partially failed run still saves evidence; scheduler can detect degraded data.
    return 0 if data["valid_quote_count"] == len(results) else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        print("配置/输入错误: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
