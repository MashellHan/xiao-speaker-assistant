"""Real-time weather query helper (wttr.in + Open-Meteo 7d forecast with 5-minute TTL cache)."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from urllib.parse import quote

import httpx

WEATHER_CITY_COORDS: dict[str, tuple[float, float]] = {
    "北京": (39.9042, 116.4074),
    "海淀": (39.9593, 116.2981),
    "朝阳区": (39.9219, 116.4436),
    "北京朝阳": (39.9219, 116.4436),
    "延庆": (40.4567, 115.9749),
    "上海": (31.2304, 121.4737),
    "广州": (23.1291, 113.2644),
    "深圳": (22.5431, 114.0579),
    "杭州": (30.2741, 120.1551),
    "南京": (32.0603, 118.7969),
    "成都": (30.5728, 104.0668),
    "重庆": (29.5630, 106.5516),
    "武汉": (30.5928, 114.3055),
    "西安": (34.3416, 108.9398),
    "天津": (39.3434, 117.3616),
    "苏州": (31.2989, 120.5853),
    "长沙": (28.2282, 112.9388),
    "郑州": (34.7466, 113.6253),
    "青岛": (36.0671, 120.3826),
    "厦门": (24.4798, 118.0894),
    "三亚": (18.2528, 109.5119),
}


async def fetch_open_meteo_7d(
    client: httpx.AsyncClient,
    city: str,
    *,
    timezone: str = "auto",
) -> list[dict[str, Any]]:
    clean = city.strip().rstrip("市区县")
    coords = WEATHER_CITY_COORDS.get(city.strip()) or WEATHER_CITY_COORDS.get(clean)
    if coords is not None:
        lat, lon = coords
    else:
        geo = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "zh"},
        )
        geo.raise_for_status()
        results = geo.json().get("results") or []
        if not results:
            return []
        lat = results[0]["latitude"]
        lon = results[0]["longitude"]
    fc = await client.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min",
            "timezone": timezone or "auto",
        },
    )
    fc.raise_for_status()
    daily = fc.json().get("daily") or {}
    dates = daily.get("time") or []
    mins = daily.get("temperature_2m_min") or []
    maxs = daily.get("temperature_2m_max") or []
    return [
        {"date": d, "min_c": lo, "max_c": hi}
        for d, lo, hi in zip(dates, mins, maxs)
    ]


async def query_city_weather(
    city: str,
    *,
    timezone: str = "auto",
    client: httpx.AsyncClient | None = None,
    cache: dict[str, tuple[float, dict[str, Any]]] | None = None,
    ttl_sec: float = 300.0,
) -> dict[str, Any]:
    """Query current weather + 7-day forecast for `city` with optional TTL caching."""
    cache_key = (city.rstrip("市区县") or city).lower()
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None and (time.monotonic() - cached[0]) < ttl_sec:
            return dict(cached[1])

    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=6.0, follow_redirects=True)
    try:
        wttr_task = http.get(
            f"https://wttr.in/{quote(city, safe='')}",
            params={"format": "j1", "lang": "zh"},
            headers={"User-Agent": "curl/8.5.0"},
        )
        meteo_task = fetch_open_meteo_7d(http, city, timezone=timezone)
        wttr_res, meteo_res = await asyncio.gather(
            wttr_task, meteo_task, return_exceptions=True
        )

        forecast_by_date: dict[str, dict[str, Any]] = {}
        if isinstance(meteo_res, list):
            for day in meteo_res:
                d_str = str(day.get("date") or "")
                if d_str:
                    forecast_by_date[d_str] = {
                        "date": d_str,
                        "min_c": day.get("min_c"),
                        "max_c": day.get("max_c"),
                    }

        result: dict[str, Any] = {"city": city}
        if isinstance(wttr_res, httpx.Response) and wttr_res.status_code < 400:
            data = wttr_res.json()
            cur = (data.get("current_condition") or [{}])[0]
            weather_days = data.get("weather") or []
            lang_desc = (cur.get("lang_zh") or cur.get("weatherDesc") or [{}])[0].get(
                "value", ""
            )
            result.update(
                {
                    "current_temp_c": cur.get("temp_C"),
                    "feels_like_c": cur.get("FeelsLikeC"),
                    "condition": lang_desc,
                    "humidity_pct": cur.get("humidity"),
                    "wind_kmph": cur.get("windspeedKmph"),
                }
            )
            for day in weather_days:
                d_str = str(day.get("date") or "")
                hourly = day.get("hourly") or []
                mid = hourly[len(hourly) // 2] if hourly else {}
                cond = (mid.get("lang_zh") or mid.get("weatherDesc") or [{}])[0].get(
                    "value", ""
                )
                entry = forecast_by_date.setdefault(d_str, {"date": d_str})
                entry.setdefault("min_c", day.get("mintempC"))
                entry.setdefault("max_c", day.get("maxtempC"))
                if cond:
                    entry["condition"] = cond

        if not forecast_by_date and isinstance(wttr_res, Exception):
            raise wttr_res

        ordered = [forecast_by_date[k] for k in sorted(forecast_by_date)]
        result["forecast"] = ordered
        if len(ordered) > 0:
            result["today_min_c"] = ordered[0].get("min_c")
            result["today_max_c"] = ordered[0].get("max_c")
        if len(ordered) > 1:
            result["tomorrow_min_c"] = ordered[1].get("min_c")
            result["tomorrow_max_c"] = ordered[1].get("max_c")

        if cache is not None and result.get("current_temp_c") is not None and ordered:
            if len(cache) >= 64:
                cache.clear()
            cache[cache_key] = (time.monotonic(), dict(result))
        return result
    finally:
        if owns_client:
            await http.aclose()
