# -*- coding: utf-8 -*-
"""
TMV Inbound Milk Run Simulation Decision Support Tool
=====================================================

App Streamlit mô phỏng và đánh giá kế hoạch Milk Run inbound cho Toyota case 1.
Thiết kế ưu tiên simulation + visualization:
- Đọc dữ liệu Toyota gốc làm baseline input.
- Tối ưu tuyến bằng heuristic Clarke-Wright Savings có ràng buộc capacity/time window.
- Chạy Monte Carlo distribution-based simulation cho travel time, loading time, dock congestion.
- Hiển thị dashboard màu định hướng quyết định: xanh/vàng/đỏ.

Chạy local:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =============================
# 0) Cấu hình hiển thị
# =============================

st.set_page_config(
    page_title="TMV Milk Run Simulator",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

STATUS_COLORS = {
    "good": "#1E9E60",      # green
    "warn": "#F2B705",      # yellow
    "bad": "#D71920",       # Toyota red
    "info": "#1F77B4",      # blue
    "muted": "#6B7280",     # gray
}

PHASE_COLORS = {
    "Driving": "#2F80ED",
    "Waiting": "#F2B705",
    "Loading": "#7B61FF",
    "Dock": "#00A6A6",
    "Delay": "#D71920",
}

DEFAULT_TMV_COORD = (21.3160, 105.6060)  # Phúc Thắng / Phúc Yên proxy coordinate
DEFAULT_EMISSION_FACTORS = {
    "Xe tải 5 tấn": 0.58,
    "Xe tải 7 tấn": 0.76,
    "Xe tải 10 tấn": 1.05,
}


# =============================
# 1) Data classes
# =============================

@dataclass(frozen=True)
class ScenarioParams:
    demand_multiplier: float = 1.0
    fuel_multiplier: float = 1.0
    road_factor: float = 1.35
    base_speed_kmh: float = 45.0
    start_min: int = 6 * 60
    tmv_end_min: int = 22 * 60
    dock_transfer_min: float = 10.0
    unload_min_per_package: float = 0.80
    overtime_penalty_vnd_per_min: float = 20_000.0
    co2_price_vnd_per_kg: float = 0.0
    n_replications: int = 200
    random_seed: int = 42
    travel_triangular: Tuple[float, float, float] = (0.95, 1.05, 1.18)
    loading_triangular: Tuple[float, float, float] = (0.90, 1.05, 1.20)
    dock_a_wait_triangular: Tuple[float, float, float] = (0.0, 8.0, 25.0)
    dock_w_wait_triangular: Tuple[float, float, float] = (0.0, 8.0, 25.0)
    unavailable_trucks: Tuple[str, ...] = tuple()
    use_stacking_adjustment: bool = True


# =============================
# 2) Helper functions
# =============================

def normalize_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_supplier_id(x) -> str:
    s = normalize_text(x).upper()
    m = re.search(r"S\d+", s)
    return m.group(0) if m else s


def normalize_truck_type(x) -> str:
    s = normalize_text(x)
    s = re.sub(r"\s+", " ", s)
    s = s.replace("Tấn", "tấn")
    return s


def normalize_packing_type(x) -> str:
    s = normalize_text(x).upper()
    if "PAL" in s:
        return "PALLET"
    if "DOL" in s:
        return "DOLLY"
    if "MOD" in s:
        return "MODULE"
    return s if s else "UNKNOWN"


def to_float(x, default: float = 0.0) -> float:
    if pd.isna(x):
        return default
    if isinstance(x, (int, float, np.number)):
        if np.isnan(float(x)):
            return default
        return float(x)
    s = str(x).strip().replace(",", "")
    if not s:
        return default
    try:
        return float(s)
    except Exception:
        return default


def parse_working_hours(text: str, default_start: int = 6 * 60, default_end: int = 22 * 60) -> Tuple[int, int]:
    s = normalize_text(text)
    matches = re.findall(r"(\d{1,2})\s*[:hH]\s*(\d{0,2})", s)
    if len(matches) >= 2:
        h1, m1 = int(matches[0][0]), int(matches[0][1] or 0)
        h2, m2 = int(matches[1][0]), int(matches[1][1] or 0)
        return h1 * 60 + m1, h2 * 60 + m2
    nums = re.findall(r"\d{1,2}", s)
    if len(nums) >= 2:
        return int(nums[0]) * 60, int(nums[1]) * 60
    return default_start, default_end


def clock(minute: float) -> str:
    if pd.isna(minute):
        return ""
    minute = int(round(float(minute)))
    h = (minute // 60) % 24
    m = minute % 60
    return f"{h:02d}:{m:02d}"


def fmt_vnd(x: float) -> str:
    try:
        return f"{float(x):,.0f} ₫"
    except Exception:
        return "0 ₫"


def fmt_pct(x: float) -> str:
    try:
        return f"{float(x) * 100:,.1f}%"
    except Exception:
        return "0.0%"


def risk_status(value: float, warn: float, bad: float, higher_is_worse: bool = True) -> str:
    if higher_is_worse:
        if value >= bad:
            return "bad"
        if value >= warn:
            return "warn"
        return "good"
    if value <= bad:
        return "bad"
    if value <= warn:
        return "warn"
    return "good"


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def deterministic_jitter(key: str, scale: float = 0.018) -> Tuple[float, float]:
    # Không dùng random global để kết quả ổn định giữa các lần chạy.
    seed = sum(ord(c) for c in key) % 10_000
    rng = np.random.default_rng(seed)
    return float(rng.normal(0, scale)), float(rng.normal(0, scale))


# =============================
# 3) Đọc dữ liệu Toyota case 1
# =============================

@st.cache_data(show_spinner=False)
def read_excel_bytes(uploaded_bytes: Optional[bytes], fallback_path: Optional[str]) -> bytes:
    if uploaded_bytes is not None:
        return uploaded_bytes
    if fallback_path and Path(fallback_path).exists():
        return Path(fallback_path).read_bytes()
    raise FileNotFoundError("Chưa có file dữ liệu Toyota. Hãy upload file Excel case 1.")


def _read_sheet(source: bytes, sheet_name: str, header=None) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(source), sheet_name=sheet_name, header=header)


@st.cache_data(show_spinner="Đang đọc dữ liệu Toyota...")
def load_toyota_case1(source: bytes) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    xl = pd.ExcelFile(io.BytesIO(source))
    sheet_names = xl.sheet_names

    def find_sheet(keyword: str) -> str:
        for s in sheet_names:
            if keyword.lower() in s.lower():
                return s
        return sheet_names[0]

    supplier_sheet = find_sheet("Danh sách")
    demand_sheet = find_sheet("Lượng hàng")
    transport_sheet = find_sheet("Thông số")

    # Supplier master
    raw_sup = pd.read_excel(io.BytesIO(source), sheet_name=supplier_sheet, header=None)
    header_idx_candidates = raw_sup.index[raw_sup.apply(lambda r: r.astype(str).str.contains("Nhà cung cấp", case=False, na=False).any(), axis=1)].tolist()
    header_idx = header_idx_candidates[0] if header_idx_candidates else 4
    sup = raw_sup.iloc[header_idx + 1 :, :3].copy()
    sup.columns = ["supplier_id", "address", "working_hours"]
    sup = sup.dropna(subset=["supplier_id"])
    sup["supplier_id"] = sup["supplier_id"].map(normalize_supplier_id)
    sup = sup[sup["supplier_id"].str.match(r"S\d+", na=False)].copy()
    sup["address"] = sup["address"].map(normalize_text)
    sup["working_hours"] = sup["working_hours"].map(normalize_text)
    windows = sup["working_hours"].apply(parse_working_hours)
    sup["window_start_min"] = [w[0] for w in windows]
    sup["window_end_min"] = [w[1] for w in windows]

    # Demand detail
    raw_dem = pd.read_excel(io.BytesIO(source), sheet_name=demand_sheet, header=None)
    # Dữ liệu thật bắt đầu sau 2 dòng header; phát hiện dòng có No. + Ngày cần nhận hàng.
    start_candidates = raw_dem.index[raw_dem.apply(lambda r: r.astype(str).str.contains("Ngày cần nhận", case=False, na=False).any(), axis=1)].tolist()
    data_start = (start_candidates[0] + 2) if start_candidates else 6
    dem = raw_dem.iloc[data_start:, :11].copy()
    dem.columns = [
        "no", "date", "supplier_id", "package_id", "quantity", "packing_type",
        "length_m", "width_m", "height_m", "dock", "remark"
    ]
    dem = dem.dropna(subset=["date", "supplier_id", "package_id"])
    dem["date"] = pd.to_datetime(dem["date"], errors="coerce").dt.date
    dem = dem.dropna(subset=["date"])
    dem["supplier_id"] = dem["supplier_id"].map(normalize_supplier_id)
    dem["packing_type"] = dem["packing_type"].map(normalize_packing_type)
    dem["dock"] = dem["dock"].map(lambda x: normalize_text(x).replace("dock", "Dock").replace("DOCK", "Dock"))
    for col in ["quantity", "length_m", "width_m", "height_m"]:
        dem[col] = dem[col].map(to_float)
    dem["raw_volume_m3"] = dem["quantity"] * dem["length_m"] * dem["width_m"] * dem["height_m"]

    # Service time table nằm trong cùng sheet demand, cột Packing type / Loading lead time.
    service_time = {"PALLET": 2.68, "DOLLY": 2.04, "MODULE": 2.04, "UNKNOWN": 2.20}
    for _, row in raw_dem.iterrows():
        p = normalize_packing_type(row.get(13, "")) if len(row) > 13 else ""
        v = to_float(row.get(14, np.nan), default=np.nan) if len(row) > 14 else np.nan
        if p in {"PALLET", "DOLLY", "MODULE"} and not pd.isna(v) and v > 0:
            service_time[p] = float(v)
    dem["load_min"] = dem.apply(lambda r: r["quantity"] * service_time.get(r["packing_type"], service_time["UNKNOWN"]), axis=1)

    # Vehicle specs and rate card
    raw_tr = pd.read_excel(io.BytesIO(source), sheet_name=transport_sheet, header=None)
    vehicle_rows = raw_tr[raw_tr.iloc[:, 1].astype(str).str.contains("Xe tải", case=False, na=False)].copy()
    vehicle_rows = vehicle_rows.iloc[:3, :5]
    trucks = vehicle_rows[[1, 2, 3, 4]].copy()
    trucks.columns = ["truck_type", "length_m", "width_m", "height_m"]
    trucks["truck_type"] = trucks["truck_type"].map(normalize_truck_type)
    for c in ["length_m", "width_m", "height_m"]:
        trucks[c] = trucks[c].map(to_float)
    trucks["capacity_m3"] = trucks["length_m"] * trucks["width_m"] * trucks["height_m"]
    trucks["emission_kg_per_km"] = trucks["truck_type"].map(DEFAULT_EMISSION_FACTORS).fillna(0.80)

    # Cost rows: cột B truck, C km range, E km rate, G fee/stop, I fee/hour
    rate_rows = raw_tr[raw_tr.iloc[:, 1].astype(str).str.contains("Xe tải", case=False, na=False)].copy()
    rate_rows = rate_rows.iloc[3:, :10]
    rates = rate_rows[[1, 2, 4, 6, 8]].copy()
    rates.columns = ["truck_type", "km_range", "km_rate_vnd", "load_wait_fee_vnd_per_stop", "tmv_wait_fee_vnd_per_hour"]
    rates["truck_type"] = rates["truck_type"].map(normalize_truck_type)
    rates = rates[rates["truck_type"].str.contains("Xe tải", na=False)].copy()
    for c in ["km_rate_vnd", "load_wait_fee_vnd_per_stop", "tmv_wait_fee_vnd_per_hour"]:
        rates[c] = rates[c].map(to_float)
    if rates.empty:
        # Fallback an toàn nếu layout file khác.
        rates = pd.DataFrame([
            ["Xe tải 5 tấn", "0-100", 14429, 17500, 60000],
            ["Xe tải 5 tấn", "101-500", 14286.5, 17500, 60000],
            ["Xe tải 5 tấn", "Từ 501 km trở lên", 13249, 17500, 60000],
            ["Xe tải 7 tấn", "0-100", 16030.5, 10000, 65000],
            ["Xe tải 7 tấn", "101-500", 15605, 10000, 65000],
            ["Xe tải 7 tấn", "Từ 501 km trở lên", 13499.5, 10000, 65000],
            ["Xe tải 10 tấn", "0-100", 25496, 10000, 75000],
            ["Xe tải 10 tấn", "101-500", 20980.5, 10000, 75000],
            ["Xe tải 10 tấn", "Từ 501 km trở lên", 16781, 10000, 75000],
        ], columns=["truck_type", "km_range", "km_rate_vnd", "load_wait_fee_vnd_per_stop", "tmv_wait_fee_vnd_per_hour"])

    return sup.reset_index(drop=True), dem.reset_index(drop=True), trucks.reset_index(drop=True), rates.reset_index(drop=True), service_time


# =============================
# 4) Stacking + daily demand aggregation
# =============================

def apply_simple_stacking_adjustment(demand: pd.DataFrame) -> pd.DataFrame:
    """Ước tính effective volume khi có stacking.

    Lưu ý: dữ liệu case dùng m3 làm capacity chính. Về vật lý, stacking chủ yếu giảm footprint
    hơn là giảm tổng m3. Hàm này chỉ áp dụng adjustment rất bảo thủ cho trường hợp đặc biệt S35
    và không tự tạo giả định quá mạnh.
    """
    df = demand.copy()
    df["effective_volume_m3"] = df["raw_volume_m3"]

    # S35 special case: Dolly 1.43 x 1.20 x 1.13 có thể chồng với 1.36 x 1.06 x 1.20,
    # chiều cao stacked effective = 2.30m thay vì 2.33m. Đây là saving nhỏ nhưng phản ánh rule case.
    mask_a = (
        (df["supplier_id"] == "S35") &
        (df["packing_type"] == "DOLLY") &
        (np.isclose(df["length_m"], 1.43, atol=0.03)) &
        (np.isclose(df["width_m"], 1.20, atol=0.03)) &
        (np.isclose(df["height_m"], 1.13, atol=0.04))
    )
    mask_b = (
        (df["supplier_id"] == "S35") &
        (df["packing_type"] == "DOLLY") &
        (np.isclose(df["length_m"], 1.36, atol=0.03)) &
        (np.isclose(df["width_m"], 1.06, atol=0.04)) &
        (np.isclose(df["height_m"], 1.20, atol=0.04))
    )
    # Chỉ giảm phần chênh cao 0.03m * max footprint cho số cặp có thể ghép.
    for (dt, sid, dock), g in df[mask_a | mask_b].groupby(["date", "supplier_id", "dock"]):
        idx_a = g[mask_a.loc[g.index]].index.tolist()
        idx_b = g[mask_b.loc[g.index]].index.tolist()
        pairs = int(min(len(idx_a), len(idx_b)))
        if pairs <= 0:
            continue
        saving_per_pair = max(1.43, 1.36) * max(1.20, 1.06) * max((1.13 + 1.20 - 2.30), 0)
        if saving_per_pair <= 0:
            continue
        for idx in idx_b[:pairs]:
            df.at[idx, "effective_volume_m3"] = max(df.at[idx, "effective_volume_m3"] - saving_per_pair, 0.01)
    return df


def build_daily_nodes(
    demand: pd.DataFrame,
    suppliers: pd.DataFrame,
    selected_date,
    trucks: pd.DataFrame,
    params: ScenarioParams,
) -> pd.DataFrame:
    d = demand[demand["date"] == selected_date].copy()
    if d.empty:
        return pd.DataFrame()
    if params.use_stacking_adjustment:
        d = apply_simple_stacking_adjustment(d)
    else:
        d["effective_volume_m3"] = d["raw_volume_m3"]

    # Scenario demand multiplier: scale volume and service time only; supplier master remains unchanged.
    d["effective_volume_m3"] *= params.demand_multiplier
    d["raw_volume_m3"] *= params.demand_multiplier
    d["load_min"] *= params.demand_multiplier
    d["quantity_scaled"] = d["quantity"] * params.demand_multiplier

    agg = d.groupby("supplier_id").agg(
        volume_m3=("effective_volume_m3", "sum"),
        raw_volume_m3=("raw_volume_m3", "sum"),
        package_count=("quantity_scaled", "sum"),
        load_min=("load_min", "sum"),
        docks=("dock", lambda s: ", ".join(sorted(set(x for x in s if x))))
    ).reset_index()

    agg = agg.merge(suppliers[["supplier_id", "address", "window_start_min", "window_end_min"]], on="supplier_id", how="left")
    agg["window_start_min"] = agg["window_start_min"].fillna(params.start_min).astype(int)
    agg["window_end_min"] = agg["window_end_min"].fillna(params.tmv_end_min).astype(int)

    max_cap = trucks[~trucks["truck_type"].isin(params.unavailable_trucks)]["capacity_m3"].max()
    max_cap = float(max_cap) if not pd.isna(max_cap) and max_cap > 0 else float(trucks["capacity_m3"].max())
    split_rows = []
    for _, r in agg.iterrows():
        n = int(math.ceil(float(r["volume_m3"]) / (max_cap * 0.96))) if max_cap > 0 else 1
        n = max(n, 1)
        for k in range(n):
            row = r.copy()
            row["node_id"] = r["supplier_id"] if n == 1 else f"{r['supplier_id']}#{k+1}"
            row["base_supplier_id"] = r["supplier_id"]
            row["volume_m3"] = float(r["volume_m3"]) / n
            row["raw_volume_m3"] = float(r["raw_volume_m3"]) / n
            row["package_count"] = float(r["package_count"]) / n
            row["load_min"] = float(r["load_min"]) / n
            split_rows.append(row)
    nodes = pd.DataFrame(split_rows)
    return nodes.reset_index(drop=True)


# =============================
# 5) Coordinates + distance/time matrix
# =============================

def infer_coord_from_address(address: str, supplier_id: str) -> Tuple[float, float, str]:
    a = normalize_text(address).lower()
    # Proxy coordinates theo khu vực, chỉ dùng khi chưa upload matrix/lat-long chính thức.
    rules = [
        (("hải phòng", "hai phong", "tràng duệ", "trang due", "hồng an"), (20.8449, 106.6881, "Hải Phòng proxy")),
        (("bắc ninh", "bac ninh", "quang châu", "quang chau"), (21.1861, 106.0763, "Bắc Ninh proxy")),
        (("hưng yên", "hung yen", "như quỳnh", "duong hao", "đường hào"), (20.9436, 106.0610, "Hưng Yên proxy")),
        (("bình xuyên", "binh xuyen", "bá thiện", "ba thien", "khai quang", "vĩnh phúc", "vinh phuc", "phú thọ", "phu tho"), (21.3100, 105.6500, "Vĩnh Phúc/Phú Thọ proxy")),
        (("quang minh", "mê linh", "me linh"), (21.1667, 105.7600, "Quang Minh/Mê Linh proxy")),
        (("sóc sơn", "soc son", "nội bài", "noi bai"), (21.2200, 105.8000, "Sóc Sơn/Nội Bài proxy")),
        (("đông anh", "dong anh"), (21.1400, 105.8300, "Đông Anh proxy")),
        (("đan phượng", "dan phuong", "phùng", "phung"), (21.1000, 105.6800, "Đan Phượng proxy")),
        (("long biên", "long bien", "sài đồng", "sai dong", "phúc lợi", "phuc loi"), (21.0550, 105.9000, "Long Biên proxy")),
        (("chương mỹ", "chuong my", "phú nghĩa", "phu nghia"), (20.9200, 105.6500, "Chương Mỹ proxy")),
        (("kim mã", "kim ma", "giảng võ", "giang vo", "đầm trấu", "dam trau", "hồng hà", "hong ha"), (21.0300, 105.8400, "Hà Nội nội đô proxy")),
    ]
    for keys, (lat, lon, label) in rules:
        if any(k in a for k in keys):
            j1, j2 = deterministic_jitter(supplier_id)
            return lat + j1, lon + j2, label
    j1, j2 = deterministic_jitter(supplier_id)
    return 21.0300 + j1, 105.8400 + j2, "Fallback Hà Nội proxy"


def add_proxy_coordinates(suppliers: pd.DataFrame) -> pd.DataFrame:
    df = suppliers.copy()
    coords = df.apply(lambda r: infer_coord_from_address(r["address"], r["supplier_id"]), axis=1)
    df["lat"] = [c[0] for c in coords]
    df["lon"] = [c[1] for c in coords]
    df["coord_source"] = [c[2] for c in coords]
    return df


def build_fallback_matrix(suppliers: pd.DataFrame, params: ScenarioParams) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], bool]:
    coords = {"TMV": DEFAULT_TMV_COORD}
    for _, r in suppliers.iterrows():
        coords[r["supplier_id"]] = (float(r["lat"]), float(r["lon"]))
    distance = {}
    duration = {}
    keys = list(coords.keys())
    for i in keys:
        for j in keys:
            if i == j:
                distance[(i, j)] = 0.0
                duration[(i, j)] = 0.0
            else:
                lat1, lon1 = coords[i]
                lat2, lon2 = coords[j]
                km = haversine_km(lat1, lon1, lat2, lon2) * params.road_factor
                # add small handling/access time, especially for industrial parks/urban links
                minutes = (km / max(params.base_speed_kmh, 1)) * 60 + min(12.0, max(3.0, km * 0.08))
                distance[(i, j)] = float(km)
                duration[(i, j)] = float(minutes)
    return distance, duration, True


def parse_uploaded_matrix(matrix_bytes: Optional[bytes]) -> Tuple[Optional[Dict[Tuple[str, str], float]], Optional[Dict[Tuple[str, str], float]], bool]:
    """Nhận file matrix optional.

    Hỗ trợ 2 dạng:
    1) Long format: from, to, distance_km, duration_min
    2) Wide format: sheet distance + sheet duration, cột/row là node codes.
    """
    if matrix_bytes is None:
        return None, None, False
    try:
        xl = pd.ExcelFile(io.BytesIO(matrix_bytes))
    except Exception:
        return None, None, False

    distance, duration = {}, {}

    # Long format: tìm trong tất cả sheet.
    for s in xl.sheet_names:
        try:
            df = pd.read_excel(io.BytesIO(matrix_bytes), sheet_name=s)
            cols = {str(c).strip().lower(): c for c in df.columns}
            from_col = next((cols[c] for c in cols if c in {"from", "origin", "i", "source"} or "from" in c or "origin" in c), None)
            to_col = next((cols[c] for c in cols if c in {"to", "destination", "j"} or "to" == c or "destination" in c), None)
            dist_col = next((cols[c] for c in cols if "distance" in c or "km" in c or "quãng" in c), None)
            dur_col = next((cols[c] for c in cols if "duration" in c or "time" in c or "minute" in c or "phút" in c), None)
            if from_col and to_col and (dist_col or dur_col):
                for _, r in df.iterrows():
                    a = normalize_supplier_id(r[from_col]) if normalize_text(r[from_col]).upper().startswith("S") else normalize_text(r[from_col]).upper()
                    b = normalize_supplier_id(r[to_col]) if normalize_text(r[to_col]).upper().startswith("S") else normalize_text(r[to_col]).upper()
                    a = "TMV" if a in {"S0", "DEPOT", "TOYOTA", "TMV"} else a
                    b = "TMV" if b in {"S0", "DEPOT", "TOYOTA", "TMV"} else b
                    if dist_col:
                        distance[(a, b)] = to_float(r[dist_col], default=np.nan)
                    if dur_col:
                        duration[(a, b)] = to_float(r[dur_col], default=np.nan)
                return distance or None, duration or None, True
        except Exception:
            continue

    # Wide matrix: sheet name contains distance / time.
    def read_wide(sheet_key: str) -> Optional[Dict[Tuple[str, str], float]]:
        for s in xl.sheet_names:
            if sheet_key in s.lower():
                df = pd.read_excel(io.BytesIO(matrix_bytes), sheet_name=s, index_col=0)
                mat = {}
                for i in df.index:
                    a = normalize_supplier_id(i) if str(i).upper().startswith("S") else str(i).strip().upper()
                    a = "TMV" if a in {"S0", "DEPOT", "TOYOTA", "TMV"} else a
                    for j in df.columns:
                        b = normalize_supplier_id(j) if str(j).upper().startswith("S") else str(j).strip().upper()
                        b = "TMV" if b in {"S0", "DEPOT", "TOYOTA", "TMV"} else b
                        mat[(a, b)] = to_float(df.loc[i, j], default=np.nan)
                return mat
        return None

    dist_wide = read_wide("distance") or read_wide("khoang") or read_wide("km")
    dur_wide = read_wide("duration") or read_wide("time") or read_wide("thoi")
    if dist_wide or dur_wide:
        return dist_wide, dur_wide, True
    return None, None, False


def get_matrix_value(mat: Dict[Tuple[str, str], float], a: str, b: str, default: float = 0.0) -> float:
    a = "TMV" if a in {"S0", "DEPOT", "TOYOTA"} else a
    b = "TMV" if b in {"S0", "DEPOT", "TOYOTA"} else b
    if (a, b) in mat and not pd.isna(mat[(a, b)]):
        return float(mat[(a, b)])
    if (b, a) in mat and not pd.isna(mat[(b, a)]):
        return float(mat[(b, a)])
    return default


# =============================
# 6) Cost + route simulation
# =============================

def km_rate_for(rates: pd.DataFrame, truck_type: str, km: float) -> float:
    r = rates[rates["truck_type"] == truck_type].copy()
    if r.empty:
        return float(rates["km_rate_vnd"].median()) if not rates.empty else 15000.0
    if km <= 100:
        cand = r[r["km_range"].astype(str).str.contains("0-100", na=False)]
    elif km <= 500:
        cand = r[r["km_range"].astype(str).str.contains("101-500", na=False)]
    else:
        cand = r[r["km_range"].astype(str).str.contains("501|trở lên|Từ", case=False, na=False)]
    if cand.empty:
        return float(r["km_rate_vnd"].median())
    return float(cand["km_rate_vnd"].iloc[0])


def stop_fee_for(rates: pd.DataFrame, truck_type: str) -> float:
    r = rates[rates["truck_type"] == truck_type]
    return float(r["load_wait_fee_vnd_per_stop"].median()) if not r.empty else 10000.0


def wait_fee_for(rates: pd.DataFrame, truck_type: str) -> float:
    r = rates[rates["truck_type"] == truck_type]
    return float(r["tmv_wait_fee_vnd_per_hour"].median()) if not r.empty else 60000.0


def select_best_truck(trucks: pd.DataFrame, rates: pd.DataFrame, volume_m3: float, distance_km: float, unavailable: Iterable[str]) -> Optional[pd.Series]:
    available = trucks[~trucks["truck_type"].isin(set(unavailable))].copy()
    feasible = available[available["capacity_m3"] >= volume_m3].copy()
    if feasible.empty:
        return None
    # Chọn truck có estimated variable cost thấp nhất, không chỉ nhỏ nhất.
    feasible["est_cost"] = feasible["truck_type"].apply(lambda t: km_rate_for(rates, t, distance_km) * distance_km)
    feasible = feasible.sort_values(["est_cost", "capacity_m3"])
    return feasible.iloc[0]


def route_distance_duration(
    sequence: List[str],
    node_to_supplier: Dict[str, str],
    dist: Dict[Tuple[str, str], float],
    dur: Dict[Tuple[str, str], float],
) -> Tuple[float, float]:
    prev = "TMV"
    total_km, total_min = 0.0, 0.0
    for node in sequence:
        sid = node_to_supplier[node]
        total_km += get_matrix_value(dist, prev, sid, 0.0)
        total_min += get_matrix_value(dur, prev, sid, 0.0)
        prev = sid
    total_km += get_matrix_value(dist, prev, "TMV", 0.0)
    total_min += get_matrix_value(dur, prev, "TMV", 0.0)
    return total_km, total_min


def simulate_route(
    route_id: str,
    sequence: List[str],
    nodes: pd.DataFrame,
    trucks: pd.DataFrame,
    rates: pd.DataFrame,
    dist: Dict[Tuple[str, str], float],
    dur: Dict[Tuple[str, str], float],
    params: ScenarioParams,
    stochastic: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    node_rows = nodes.set_index("node_id")
    node_to_supplier = nodes.set_index("node_id")["base_supplier_id"].to_dict()
    total_volume = float(sum(node_rows.loc[n, "volume_m3"] for n in sequence))
    total_packages = float(sum(node_rows.loc[n, "package_count"] for n in sequence))
    planned_km, _ = route_distance_duration(sequence, node_to_supplier, dist, dur)
    truck = select_best_truck(trucks, rates, total_volume, planned_km, params.unavailable_trucks)
    if truck is None:
        # fallback dùng truck lớn nhất để vẫn show infeasible route.
        truck = trucks.sort_values("capacity_m3", ascending=False).iloc[0]
    truck_type = truck["truck_type"]
    capacity = float(truck["capacity_m3"])
    emission_factor = float(truck.get("emission_kg_per_km", DEFAULT_EMISSION_FACTORS.get(truck_type, 0.8)))

    travel_tri = params.travel_triangular
    loading_tri = params.loading_triangular
    if rng is None:
        rng = np.random.default_rng(params.random_seed)

    def draw_tri(tri: Tuple[float, float, float]) -> float:
        if not stochastic:
            return tri[1]
        a, m, b = tri
        return float(rng.triangular(a, m, b))

    events = []
    stops = []
    time_now = float(params.start_min)
    prev = "TMV"
    total_km = 0.0
    driving_min = 0.0
    waiting_min = 0.0
    loading_min_total = 0.0
    delay_min_total = 0.0
    late_stops = 0

    for n in sequence:
        r = node_rows.loc[n]
        sid = r["base_supplier_id"]
        base_km = get_matrix_value(dist, prev, sid, 0.0)
        base_travel = get_matrix_value(dur, prev, sid, 0.0)
        travel_factor = draw_tri(travel_tri)
        travel_min = base_travel * travel_factor
        start_drive = time_now
        end_drive = time_now + travel_min
        events.append({
            "route_id": route_id, "node": sid, "phase": "Driving", "start_min": start_drive,
            "end_min": end_drive, "label": f"{prev} → {sid}", "minutes": travel_min
        })
        total_km += base_km
        driving_min += travel_min
        time_now = end_drive

        win_start = float(r["window_start_min"])
        win_end = float(r["window_end_min"])
        early_wait = max(win_start - time_now, 0.0)
        if early_wait > 0:
            events.append({
                "route_id": route_id, "node": sid, "phase": "Waiting", "start_min": time_now,
                "end_min": time_now + early_wait, "label": f"Wait for {sid} window", "minutes": early_wait
            })
            waiting_min += early_wait
            time_now += early_wait

        late = max(time_now - win_end, 0.0)
        if late > 0:
            late_stops += 1
            delay_min_total += late
            events.append({
                "route_id": route_id, "node": sid, "phase": "Delay", "start_min": win_end,
                "end_min": time_now, "label": f"Late at {sid}", "minutes": late
            })

        load_base = float(r["load_min"])
        load_min = load_base * draw_tri(loading_tri)
        events.append({
            "route_id": route_id, "node": sid, "phase": "Loading", "start_min": time_now,
            "end_min": time_now + load_min, "label": f"Load {sid}", "minutes": load_min
        })
        stop_arrival = time_now
        time_now += load_min
        loading_min_total += load_min
        stops.append({
            "route_id": route_id,
            "node_id": n,
            "supplier_id": sid,
            "planned_window": f"{clock(win_start)}-{clock(win_end)}",
            "arrival_min": stop_arrival,
            "arrival_time": clock(stop_arrival),
            "depart_min": time_now,
            "depart_time": clock(time_now),
            "late_min": late,
            "wait_min": early_wait,
            "load_min": load_min,
            "volume_m3": float(r["volume_m3"]),
            "dock": r["docks"],
            "status": "Late" if late > 0 else "On time"
        })
        prev = sid

    # Return to TMV
    base_km = get_matrix_value(dist, prev, "TMV", 0.0)
    base_travel = get_matrix_value(dur, prev, "TMV", 0.0)
    travel_min = base_travel * draw_tri(travel_tri)
    events.append({
        "route_id": route_id, "node": "TMV", "phase": "Driving", "start_min": time_now,
        "end_min": time_now + travel_min, "label": f"{prev} → TMV", "minutes": travel_min
    })
    total_km += base_km
    driving_min += travel_min
    time_now += travel_min
    arrival_tmv = time_now

    # Dock wait and unload
    route_docks = ", ".join(sorted(set(str(node_rows.loc[n, "docks"]) for n in sequence)))
    uses_a = "A" in route_docks.upper()
    uses_w = "W" in route_docks.upper()
    dock_transfer = params.dock_transfer_min if uses_a and uses_w else 0.0
    dock_wait = 0.0
    if stochastic:
        if uses_a:
            dock_wait += float(rng.triangular(*params.dock_a_wait_triangular))
        if uses_w:
            dock_wait += float(rng.triangular(*params.dock_w_wait_triangular))
        if uses_a and uses_w:
            dock_wait = dock_wait / 2
    else:
        # planned mode uses modal dock wait, not the maximum.
        vals = []
        if uses_a:
            vals.append(params.dock_a_wait_triangular[1])
        if uses_w:
            vals.append(params.dock_w_wait_triangular[1])
        dock_wait = float(np.mean(vals)) if vals else 0.0

    dock_phase_start = time_now
    dock_service = dock_wait + dock_transfer + total_packages * params.unload_min_per_package
    if dock_service > 0:
        events.append({
            "route_id": route_id, "node": "TMV", "phase": "Dock", "start_min": dock_phase_start,
            "end_min": dock_phase_start + dock_service, "label": "Dock unload / transfer", "minutes": dock_service
        })
    waiting_min += dock_wait
    time_now += dock_service
    finish_tmv = time_now
    delivery_late = max(finish_tmv - params.tmv_end_min, 0.0)
    if delivery_late > 0:
        delay_min_total += delivery_late
        events.append({
            "route_id": route_id, "node": "TMV", "phase": "Delay", "start_min": params.tmv_end_min,
            "end_min": finish_tmv, "label": "Late delivery at TMV", "minutes": delivery_late
        })

    km_rate = km_rate_for(rates, truck_type, total_km)
    stop_fee = stop_fee_for(rates, truck_type)
    wait_fee = wait_fee_for(rates, truck_type)
    transport_cost = total_km * km_rate * params.fuel_multiplier
    stop_cost = len(sequence) * stop_fee
    wait_cost = (waiting_min / 60) * wait_fee
    overtime_cost = max(finish_tmv - params.tmv_end_min, 0.0) * params.overtime_penalty_vnd_per_min
    co2_kg = total_km * emission_factor
    co2_cost = co2_kg * params.co2_price_vnd_per_kg
    total_cost = transport_cost + stop_cost + wait_cost + overtime_cost + co2_cost

    route = {
        "route_id": route_id,
        "sequence": " → ".join(node_to_supplier[n] for n in sequence),
        "nodes": sequence,
        "truck_type": truck_type,
        "capacity_m3": capacity,
        "volume_m3": total_volume,
        "utilization": total_volume / capacity if capacity else np.nan,
        "package_count": total_packages,
        "stops": len(sequence),
        "docks": route_docks,
        "distance_km": total_km,
        "driving_min": driving_min,
        "waiting_min": waiting_min,
        "loading_min": loading_min_total,
        "delay_min": delay_min_total,
        "late_stops": late_stops,
        "arrival_tmv_min": arrival_tmv,
        "finish_tmv_min": finish_tmv,
        "arrival_tmv_time": clock(arrival_tmv),
        "finish_tmv_time": clock(finish_tmv),
        "delivery_late_min": delivery_late,
        "transport_cost_vnd": transport_cost,
        "stop_cost_vnd": stop_cost,
        "wait_cost_vnd": wait_cost,
        "overtime_cost_vnd": overtime_cost,
        "co2_kg": co2_kg,
        "co2_cost_vnd": co2_cost,
        "total_cost_vnd": total_cost,
        "feasible_capacity": total_volume <= capacity + 1e-6,
        "feasible_time": late_stops == 0 and delivery_late <= 0,
        "status": "OK" if (total_volume <= capacity + 1e-6 and late_stops == 0 and delivery_late <= 0) else "Check",
    }
    return route, pd.DataFrame(stops), pd.DataFrame(events)


# =============================
# 7) Route optimizer heuristic
# =============================

def route_feasible(sequence: List[str], nodes, trucks, rates, dist, dur, params: ScenarioParams) -> bool:
    r, _, _ = simulate_route("TEST", sequence, nodes, trucks, rates, dist, dur, params, stochastic=False)
    return bool(r["feasible_capacity"] and r["feasible_time"])


def build_routes_clarke_wright(
    nodes: pd.DataFrame,
    trucks: pd.DataFrame,
    rates: pd.DataFrame,
    dist: Dict[Tuple[str, str], float],
    dur: Dict[Tuple[str, str], float],
    params: ScenarioParams,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[List[str]]]:
    if nodes.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []
    node_to_supplier = nodes.set_index("node_id")["base_supplier_id"].to_dict()
    node_volume = nodes.set_index("node_id")["volume_m3"].to_dict()
    available_trucks = trucks[~trucks["truck_type"].isin(set(params.unavailable_trucks))]
    max_cap = float(available_trucks["capacity_m3"].max()) if not available_trucks.empty else float(trucks["capacity_m3"].max())

    route_list: List[List[str]] = [[n] for n in nodes["node_id"].tolist()]

    def route_volume(seq: List[str]) -> float:
        return float(sum(node_volume[n] for n in seq))

    # Savings list
    savings = []
    ids = nodes["node_id"].tolist()
    for i in ids:
        for j in ids:
            if i == j:
                continue
            si, sj = node_to_supplier[i], node_to_supplier[j]
            sav = get_matrix_value(dist, "TMV", si, 0) + get_matrix_value(dist, "TMV", sj, 0) - get_matrix_value(dist, si, sj, 0)
            savings.append((sav, i, j))
    savings.sort(reverse=True, key=lambda x: x[0])

    def find_route_idx(node: str) -> Optional[int]:
        for idx, seq in enumerate(route_list):
            if node in seq:
                return idx
        return None

    for _, i, j in savings:
        ai, bj = find_route_idx(i), find_route_idx(j)
        if ai is None or bj is None or ai == bj:
            continue
        ra, rb = route_list[ai], route_list[bj]
        # Thử các hướng ghép đơn giản và chọn sequence feasible/cost thấp nhất.
        candidates = []
        if ra[-1] == i and rb[0] == j:
            candidates.append(ra + rb)
        if rb[-1] == j and ra[0] == i:
            candidates.append(rb + ra)
        if ra[0] == i and rb[0] == j:
            candidates.append(list(reversed(ra)) + rb)
        if ra[-1] == i and rb[-1] == j:
            candidates.append(ra + list(reversed(rb)))
        best = None
        best_cost = float("inf")
        for cand in candidates:
            if route_volume(cand) > max_cap + 1e-6:
                continue
            r, _, _ = simulate_route("TEST", cand, nodes, trucks, rates, dist, dur, params, stochastic=False)
            if r["feasible_capacity"] and r["feasible_time"] and r["total_cost_vnd"] < best_cost:
                best = cand
                best_cost = r["total_cost_vnd"]
        if best is None:
            continue
        # Merge
        for idx in sorted([ai, bj], reverse=True):
            route_list.pop(idx)
        route_list.append(best)

    # Final simulate deterministic plan
    route_rows = []
    stop_frames = []
    event_frames = []
    # Sort routes by finish time then ID for readable dashboard.
    temp = []
    for k, seq in enumerate(route_list, start=1):
        rid = f"R{k:02d}"
        r, stops, events = simulate_route(rid, seq, nodes, trucks, rates, dist, dur, params, stochastic=False)
        temp.append((r["finish_tmv_min"], rid, seq, r, stops, events))
    temp.sort(key=lambda x: x[0])
    for k, (_, _, seq, _, _, _) in enumerate(temp, start=1):
        rid = f"R{k:02d}"
        r, stops, events = simulate_route(rid, seq, nodes, trucks, rates, dist, dur, params, stochastic=False)
        route_rows.append(r)
        stop_frames.append(stops)
        event_frames.append(events)
    routes_df = pd.DataFrame(route_rows)
    stops_df = pd.concat(stop_frames, ignore_index=True) if stop_frames else pd.DataFrame()
    events_df = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    return routes_df, stops_df, events_df, route_list


# =============================
# 8) Monte Carlo simulation
# =============================

def run_monte_carlo(
    routes_df: pd.DataFrame,
    nodes: pd.DataFrame,
    trucks: pd.DataFrame,
    rates: pd.DataFrame,
    dist: Dict[Tuple[str, str], float],
    dur: Dict[Tuple[str, str], float],
    params: ScenarioParams,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if routes_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    rng = np.random.default_rng(params.random_seed)
    route_reps = []
    stop_reps = []
    scenario_reps = []
    for rep in range(1, params.n_replications + 1):
        total_cost = 0.0
        total_km = 0.0
        total_co2 = 0.0
        total_delay = 0.0
        total_stops = 0
        late_stops = 0
        late_routes = 0
        for _, rr in routes_df.iterrows():
            seq = rr["nodes"]
            route, stops, _ = simulate_route(
                rr["route_id"], seq, nodes, trucks, rates, dist, dur, params,
                stochastic=True, rng=rng
            )
            route["replication"] = rep
            route["route_late_flag"] = int(route["late_stops"] > 0 or route["delivery_late_min"] > 0)
            route_reps.append(route)
            if not stops.empty:
                stops["replication"] = rep
                stop_reps.append(stops)
            total_cost += route["total_cost_vnd"]
            total_km += route["distance_km"]
            total_co2 += route["co2_kg"]
            total_delay += route["delay_min"]
            total_stops += route["stops"]
            late_stops += route["late_stops"]
            late_routes += int(route["late_stops"] > 0 or route["delivery_late_min"] > 0)
        scenario_reps.append({
            "replication": rep,
            "total_cost_vnd": total_cost,
            "distance_km": total_km,
            "co2_kg": total_co2,
            "delay_min": total_delay,
            "pickup_on_time_rate": 1 - (late_stops / max(total_stops, 1)),
            "route_late_rate": late_routes / max(len(routes_df), 1),
        })
    route_rep_df = pd.DataFrame(route_reps)
    stop_rep_df = pd.concat(stop_reps, ignore_index=True) if stop_reps else pd.DataFrame()
    scenario_rep_df = pd.DataFrame(scenario_reps)
    return route_rep_df, stop_rep_df, scenario_rep_df


def summarize_plan(routes_df: pd.DataFrame, mc_scenario: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    if routes_df.empty:
        return {}
    if mc_scenario is not None and not mc_scenario.empty:
        cost_mean = float(mc_scenario["total_cost_vnd"].mean())
        cost_p95 = float(mc_scenario["total_cost_vnd"].quantile(0.95))
        on_time = float(mc_scenario["pickup_on_time_rate"].mean())
        route_late = float(mc_scenario["route_late_rate"].mean())
        delay_mean = float(mc_scenario["delay_min"].mean())
    else:
        cost_mean = float(routes_df["total_cost_vnd"].sum())
        cost_p95 = cost_mean
        total_stops = float(routes_df["stops"].sum())
        on_time = 1 - float(routes_df["late_stops"].sum()) / max(total_stops, 1)
        route_late = float(((routes_df["late_stops"] > 0) | (routes_df["delivery_late_min"] > 0)).mean())
        delay_mean = float(routes_df["delay_min"].sum())
    return {
        "total_cost_vnd": cost_mean,
        "cost_p95_vnd": cost_p95,
        "routes": float(len(routes_df)),
        "distance_km": float(routes_df["distance_km"].sum()),
        "co2_kg": float(routes_df["co2_kg"].sum()),
        "utilization": float(routes_df["volume_m3"].sum() / routes_df["capacity_m3"].sum()),
        "pickup_on_time_rate": on_time,
        "route_late_rate": route_late,
        "delay_min": delay_mean,
        "waiting_min": float(routes_df["waiting_min"].sum()),
        "volume_m3": float(routes_df["volume_m3"].sum()),
        "capacity_m3": float(routes_df["capacity_m3"].sum()),
    }


def build_route_risk(routes_df: pd.DataFrame, route_reps: pd.DataFrame) -> pd.DataFrame:
    if routes_df.empty:
        return pd.DataFrame()
    if route_reps.empty:
        out = routes_df[["route_id", "sequence", "truck_type", "utilization", "delay_min", "total_cost_vnd"]].copy()
        out["late_probability"] = ((routes_df["late_stops"] > 0) | (routes_df["delivery_late_min"] > 0)).astype(float)
        out["p95_finish_min"] = routes_df["finish_tmv_min"]
        out["risk_status"] = out["late_probability"].apply(lambda x: risk_status(x, 0.10, 0.25))
        return out
    risk = route_reps.groupby("route_id").agg(
        late_probability=("route_late_flag", "mean"),
        avg_delay_min=("delay_min", "mean"),
        p95_finish_min=("finish_tmv_min", lambda s: s.quantile(0.95)),
        avg_cost_vnd=("total_cost_vnd", "mean"),
    ).reset_index()
    base_cols = ["route_id", "sequence", "truck_type", "utilization", "volume_m3", "capacity_m3", "distance_km", "docks"]
    out = routes_df[base_cols].merge(risk, on="route_id", how="left")
    out["p95_finish_time"] = out["p95_finish_min"].map(clock)
    out["risk_status"] = out["late_probability"].apply(lambda x: risk_status(float(x or 0), 0.10, 0.25))
    return out


# =============================
# 9) Visualization
# =============================

def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .block-container {{ padding-top: 1.4rem; }}
        .kpi-card {{
            border-radius: 18px;
            padding: 18px 18px;
            background: #FFFFFF;
            box-shadow: 0 6px 18px rgba(0,0,0,0.06);
            border-left: 8px solid #6B7280;
            min-height: 126px;
        }}
        .kpi-label {{ color: #6B7280; font-size: 0.88rem; margin-bottom: 6px; }}
        .kpi-value {{ color: #111827; font-size: 1.55rem; font-weight: 800; line-height: 1.15; }}
        .kpi-note {{ color: #6B7280; font-size: 0.80rem; margin-top: 8px; }}
        .decision-box {{
            border-radius: 16px; padding: 16px 18px; color: #111827;
            background: #fff; border: 1px solid #E5E7EB;
        }}
        .small-muted {{ color:#6B7280; font-size:0.86rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, note: str, status: str = "muted") -> None:
    color = STATUS_COLORS.get(status, STATUS_COLORS["muted"])
    st.markdown(
        f"""
        <div class="kpi-card" style="border-left-color:{color};">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def decision_recommendation(summary: Dict[str, float]) -> Tuple[str, str]:
    if not summary:
        return "info", "Chưa có kết quả mô phỏng."
    if summary["pickup_on_time_rate"] < 0.90:
        return "bad", "Rủi ro đúng giờ cao: cần tách tuyến, thêm buffer hoặc đổi sang xe lớn hơn/route ít điểm hơn."
    if summary["utilization"] < 0.75:
        return "warn", "Hiệu suất xe còn thấp: nên thử ghép tuyến hoặc dùng xe nhỏ hơn nếu time window vẫn khả thi."
    if summary["route_late_rate"] > 0.25:
        return "bad", "Nhiều tuyến có xác suất trễ cao trong Monte Carlo: cần xem tab Route Risk để xử lý tuyến đỏ."
    if summary["utilization"] >= 0.90 and summary["pickup_on_time_rate"] >= 0.95:
        return "good", "Phương án tốt: utilization cao và xác suất đúng giờ ổn định."
    return "good", "Phương án chấp nhận được: tiếp tục kiểm tra tuyến vàng trước khi vận hành."


def plot_route_map(routes_df: pd.DataFrame, suppliers_geo: pd.DataFrame, route_risk: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if routes_df.empty:
        return fig
    coords = suppliers_geo.set_index("supplier_id")[["lat", "lon"]].to_dict("index")
    risk_map = route_risk.set_index("route_id")["risk_status"].to_dict() if not route_risk.empty else {}
    for _, r in routes_df.iterrows():
        status = risk_map.get(r["route_id"], "good")
        color = STATUS_COLORS[status]
        seq_sup = [s.strip() for s in str(r["sequence"]).split("→")]
        lats = [DEFAULT_TMV_COORD[0]]
        lons = [DEFAULT_TMV_COORD[1]]
        texts = ["TMV"]
        for sid in seq_sup:
            sid = sid.strip()
            if sid in coords:
                lats.append(coords[sid]["lat"])
                lons.append(coords[sid]["lon"])
                texts.append(sid)
        lats.append(DEFAULT_TMV_COORD[0])
        lons.append(DEFAULT_TMV_COORD[1])
        texts.append("TMV")
        fig.add_trace(go.Scattermapbox(
            lat=lats, lon=lons, mode="lines+markers+text", text=texts,
            textposition="top center",
            name=f"{r['route_id']} | {r['truck_type']}",
            line=dict(width=max(2, min(7, 2 + r["volume_m3"] / 8)), color=color),
            marker=dict(size=10, color=color),
            hovertemplate="%{text}<extra></extra>",
        ))
    fig.add_trace(go.Scattermapbox(
        lat=[DEFAULT_TMV_COORD[0]], lon=[DEFAULT_TMV_COORD[1]], mode="markers+text",
        text=["TMV"], textposition="bottom right", name="TMV",
        marker=dict(size=18, color="#111827"),
    ))
    fig.update_layout(
        mapbox=dict(style="open-street-map", zoom=7.2, center=dict(lat=21.16, lon=105.95)),
        margin=dict(l=0, r=0, t=10, b=0),
        height=620,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def plot_timeline(events_df: pd.DataFrame, selected_date) -> go.Figure:
    if events_df.empty:
        return go.Figure()
    ev = events_df.copy()
    day = pd.Timestamp(selected_date)
    ev["Start"] = ev["start_min"].apply(lambda m: day + pd.Timedelta(minutes=float(m)))
    ev["Finish"] = ev["end_min"].apply(lambda m: day + pd.Timedelta(minutes=float(m)))
    ev["Task"] = ev["route_id"]
    ev["Detail"] = ev["label"] + " | " + ev["minutes"].round(1).astype(str) + " min"
    fig = px.timeline(
        ev,
        x_start="Start",
        x_end="Finish",
        y="Task",
        color="phase",
        hover_data=["Detail", "node", "minutes"],
        color_discrete_map=PHASE_COLORS,
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=max(440, 55 * ev["route_id"].nunique()), margin=dict(l=10, r=10, t=30, b=10))
    return fig


def plot_cost_waterfall(routes_df: pd.DataFrame) -> go.Figure:
    if routes_df.empty:
        return go.Figure()
    vals = [
        routes_df["transport_cost_vnd"].sum(),
        routes_df["stop_cost_vnd"].sum(),
        routes_df["wait_cost_vnd"].sum(),
        routes_df["overtime_cost_vnd"].sum(),
        routes_df["co2_cost_vnd"].sum(),
    ]
    fig = go.Figure(go.Waterfall(
        name="Cost",
        orientation="v",
        measure=["relative", "relative", "relative", "relative", "relative", "total"],
        x=["Km cost", "Stop fee", "Waiting", "Overtime", "CO₂ internal", "Total"],
        y=vals + [sum(vals)],
        connector={"line": {"color": "#9CA3AF"}},
        increasing={"marker": {"color": "#D71920"}},
        decreasing={"marker": {"color": "#1E9E60"}},
        totals={"marker": {"color": "#111827"}},
    ))
    fig.update_layout(height=430, yaxis_title="VND", margin=dict(l=10, r=10, t=30, b=10))
    return fig


def plot_scenario_box(mc_baseline: pd.DataFrame, mc_scenario: pd.DataFrame) -> go.Figure:
    if mc_baseline.empty or mc_scenario.empty:
        return go.Figure()
    df = pd.concat([
        mc_baseline.assign(scenario="Baseline"),
        mc_scenario.assign(scenario="Current scenario"),
    ], ignore_index=True)
    fig = px.box(df, x="scenario", y="total_cost_vnd", points=False, color="scenario",
                 color_discrete_map={"Baseline": "#1F77B4", "Current scenario": "#D71920"})
    fig.update_layout(height=430, yaxis_title="Total cost per replication (VND)", showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def to_excel_download(
    routes_df: pd.DataFrame,
    stops_df: pd.DataFrame,
    events_df: pd.DataFrame,
    route_risk: pd.DataFrame,
    scenario_summary: Dict[str, float],
    scenario_params: ScenarioParams,
) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        routes_export = routes_df.copy()
        if "nodes" in routes_export.columns:
            routes_export["nodes"] = routes_export["nodes"].astype(str)
        routes_export.to_excel(writer, sheet_name="route_summary", index=False)
        stops_df.to_excel(writer, sheet_name="stop_schedule", index=False)
        events_df.to_excel(writer, sheet_name="timeline_events", index=False)
        route_risk.to_excel(writer, sheet_name="route_risk", index=False)
        pd.DataFrame([scenario_summary]).to_excel(writer, sheet_name="kpi_summary", index=False)
        pd.DataFrame([scenario_params.__dict__]).to_excel(writer, sheet_name="scenario_params", index=False)
        workbook = writer.book
        fmt_header = workbook.add_format({"bold": True, "bg_color": "#D71920", "font_color": "white"})
        fmt_pct = workbook.add_format({"num_format": "0.0%"})
        fmt_money = workbook.add_format({"num_format": "#,##0"})
        for ws_name in writer.sheets:
            ws = writer.sheets[ws_name]
            ws.freeze_panes(1, 0)
            ws.set_row(0, None, fmt_header)
            ws.set_column(0, 40, 16)
        if "route_summary" in writer.sheets:
            writer.sheets["route_summary"].set_column("I:I", 12, fmt_pct)
            writer.sheets["route_summary"].set_column("X:AD", 15, fmt_money)
    return output.getvalue()


# =============================
# 10) Scenario presets
# =============================

def load_scenario_presets() -> Dict[str, dict]:
    default_path = Path(__file__).with_name("scenarios.json")
    if default_path.exists():
        try:
            return json.loads(default_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "Normal baseline": {
            "demand_multiplier": 1.0, "fuel_multiplier": 1.0,
            "dock_a_wait_mode": 8, "dock_w_wait_mode": 8,
            "travel_triangular": [0.95, 1.05, 1.18],
            "loading_triangular": [0.90, 1.05, 1.20],
            "unavailable_trucks": [],
        }
    }


def make_params_from_ui(preset: dict, ui: dict) -> ScenarioParams:
    travel = tuple(ui.get("travel_triangular") or preset.get("travel_triangular", [0.95, 1.05, 1.18]))
    loading = tuple(ui.get("loading_triangular") or preset.get("loading_triangular", [0.90, 1.05, 1.20]))
    da_mode = float(ui.get("dock_a_wait_mode", preset.get("dock_a_wait_mode", 8)))
    dw_mode = float(ui.get("dock_w_wait_mode", preset.get("dock_w_wait_mode", 8)))
    return ScenarioParams(
        demand_multiplier=float(ui.get("demand_multiplier", preset.get("demand_multiplier", 1.0))),
        fuel_multiplier=float(ui.get("fuel_multiplier", preset.get("fuel_multiplier", 1.0))),
        road_factor=float(ui.get("road_factor", 1.35)),
        base_speed_kmh=float(ui.get("base_speed_kmh", 45.0)),
        start_min=int(ui.get("start_min", 6 * 60)),
        tmv_end_min=int(ui.get("tmv_end_min", 22 * 60)),
        dock_transfer_min=float(ui.get("dock_transfer_min", 10.0)),
        unload_min_per_package=float(ui.get("unload_min_per_package", 0.80)),
        overtime_penalty_vnd_per_min=float(ui.get("overtime_penalty_vnd_per_min", 20_000)),
        co2_price_vnd_per_kg=float(ui.get("co2_price_vnd_per_kg", 0.0)),
        n_replications=int(ui.get("n_replications", 200)),
        random_seed=int(ui.get("random_seed", 42)),
        travel_triangular=(float(travel[0]), float(travel[1]), float(travel[2])),
        loading_triangular=(float(loading[0]), float(loading[1]), float(loading[2])),
        dock_a_wait_triangular=(0.0, da_mode, max(da_mode * 2.5, da_mode + 1)),
        dock_w_wait_triangular=(0.0, dw_mode, max(dw_mode * 2.5, dw_mode + 1)),
        unavailable_trucks=tuple(ui.get("unavailable_trucks", preset.get("unavailable_trucks", []))),
        use_stacking_adjustment=bool(ui.get("use_stacking_adjustment", True)),
    )


# =============================
# 11) Main App
# =============================

def run_app() -> None:
    inject_css()
    st.title("🚚 TMV Inbound Milk Run Simulator")
    st.caption("Simulation-first decision support tool: Toyota baseline data → optimized plan → Monte Carlo risk simulation → dashboard quyết định.")

    presets = load_scenario_presets()

    with st.sidebar:
        st.header("1) Dữ liệu")
        uploaded = st.file_uploader("Upload `dữ liệu case 1 toyota.xlsx`", type=["xlsx"])
        matrix_upload = st.file_uploader("Optional: upload distance/time matrix", type=["xlsx"])
        fallback_path = "/mnt/data/dữ liệu case 1 toyota.xlsx" if Path("/mnt/data/dữ liệu case 1 toyota.xlsx").exists() else None
        if uploaded is None and fallback_path:
            st.info("Đang dùng file Toyota có sẵn trong môi trường sandbox. Khi chạy máy bạn, hãy upload file Excel.")
        elif uploaded is None:
            st.warning("Hãy upload file dữ liệu Toyota case 1 để bắt đầu.")
            st.stop()

    toyota_bytes = read_excel_bytes(uploaded.getvalue() if uploaded else None, fallback_path)
    suppliers, demand, trucks, rates, service_time = load_toyota_case1(toyota_bytes)
    suppliers_geo = add_proxy_coordinates(suppliers)

    available_dates = sorted(demand["date"].dropna().unique().tolist())
    if not available_dates:
        st.error("Không tìm thấy cột ngày nhận hàng trong file demand.")
        st.stop()

    with st.sidebar:
        st.header("2) Scenario")
        selected_date = st.selectbox("Ngày mô phỏng", available_dates, index=0, format_func=lambda d: pd.Timestamp(d).strftime("%d/%m/%Y"))
        preset_name = st.selectbox("Preset scenario", list(presets.keys()), index=0)
        preset = presets[preset_name]

        st.markdown("**Distribution-based simulation**")
        n_rep = st.slider("Số lần Monte Carlo", 50, 800, int(200), step=50)
        seed = st.number_input("Random seed", value=42, min_value=1, step=1)
        demand_multiplier = st.slider("Demand multiplier", 0.70, 1.50, float(preset.get("demand_multiplier", 1.0)), step=0.05)
        fuel_multiplier = st.slider("Fuel / km cost multiplier", 0.80, 1.50, float(preset.get("fuel_multiplier", 1.0)), step=0.05)

        st.markdown("**Travel time factor: triangular(min, mode, max)**")
        default_travel = preset.get("travel_triangular", [0.95, 1.05, 1.18])
        t_min = st.slider("Travel min", 0.75, 1.50, float(default_travel[0]), step=0.01)
        t_mode = st.slider("Travel mode", 0.80, 1.80, float(default_travel[1]), step=0.01)
        t_max = st.slider("Travel max", 1.00, 2.50, float(default_travel[2]), step=0.01)

        st.markdown("**Loading time factor: triangular(min, mode, max)**")
        default_loading = preset.get("loading_triangular", [0.90, 1.05, 1.20])
        l_min = st.slider("Loading min", 0.70, 1.50, float(default_loading[0]), step=0.01)
        l_mode = st.slider("Loading mode", 0.80, 1.80, float(default_loading[1]), step=0.01)
        l_max = st.slider("Loading max", 1.00, 2.50, float(default_loading[2]), step=0.01)

        dock_a_wait_mode = st.slider("Dock A wait mode / phút", 0, 90, int(preset.get("dock_a_wait_mode", 8)), step=5)
        dock_w_wait_mode = st.slider("Dock W wait mode / phút", 0, 90, int(preset.get("dock_w_wait_mode", 8)), step=5)

        unavailable_trucks = st.multiselect(
            "Xe không khả dụng",
            options=trucks["truck_type"].tolist(),
            default=preset.get("unavailable_trucks", []),
        )

        with st.expander("Advanced assumptions"):
            road_factor = st.slider("Road distance factor cho fallback matrix", 1.00, 1.80, 1.35, step=0.05)
            base_speed = st.slider("Base speed km/h cho fallback matrix", 25, 75, 45, step=5)
            start_hour = st.slider("Giờ xuất phát giả định từ TMV", 0, 12, 6)
            tmv_end_hour = st.slider("TMV receiving end hour", 12, 24, 22)
            unload_min_per_package = st.slider("Unload minute / package tại TMV", 0.2, 3.0, 0.8, step=0.1)
            overtime_penalty = st.number_input("Overtime penalty VND/min", value=20_000, min_value=0, step=5_000)
            co2_price = st.number_input("Internal CO₂ price VND/kg", value=0, min_value=0, step=100)
            use_stacking = st.checkbox("Áp dụng S35 stacking adjustment bảo thủ", value=True)

        run_btn = st.button("▶️ Run simulation", type="primary", use_container_width=True)

    ui = {
        "demand_multiplier": demand_multiplier,
        "fuel_multiplier": fuel_multiplier,
        "travel_triangular": (t_min, max(t_mode, t_min), max(t_max, t_mode)),
        "loading_triangular": (l_min, max(l_mode, l_min), max(l_max, l_mode)),
        "dock_a_wait_mode": dock_a_wait_mode,
        "dock_w_wait_mode": dock_w_wait_mode,
        "unavailable_trucks": unavailable_trucks,
        "n_replications": n_rep,
        "random_seed": seed,
        "road_factor": road_factor,
        "base_speed_kmh": base_speed,
        "start_min": start_hour * 60,
        "tmv_end_min": tmv_end_hour * 60,
        "unload_min_per_package": unload_min_per_package,
        "overtime_penalty_vnd_per_min": overtime_penalty,
        "co2_price_vnd_per_kg": co2_price,
        "use_stacking_adjustment": use_stacking,
    }
    scenario_params = make_params_from_ui(preset, ui)
    baseline_params = ScenarioParams(
        n_replications=scenario_params.n_replications,
        random_seed=scenario_params.random_seed,
        road_factor=scenario_params.road_factor,
        base_speed_kmh=scenario_params.base_speed_kmh,
        start_min=scenario_params.start_min,
        tmv_end_min=scenario_params.tmv_end_min,
        dock_transfer_min=scenario_params.dock_transfer_min,
        unload_min_per_package=scenario_params.unload_min_per_package,
        overtime_penalty_vnd_per_min=scenario_params.overtime_penalty_vnd_per_min,
        co2_price_vnd_per_kg=scenario_params.co2_price_vnd_per_kg,
        unavailable_trucks=tuple(),
        use_stacking_adjustment=scenario_params.use_stacking_adjustment,
    )

    matrix_bytes = matrix_upload.getvalue() if matrix_upload else None
    uploaded_dist, uploaded_dur, matrix_ok = parse_uploaded_matrix(matrix_bytes)
    fallback_dist, fallback_dur, used_fallback = build_fallback_matrix(suppliers_geo, scenario_params)
    dist = uploaded_dist or fallback_dist
    dur = uploaded_dur or fallback_dur
    # Nếu chỉ có distance mà không có duration, suy ra duration; nếu chỉ có duration thì distance fallback.
    if uploaded_dist and not uploaded_dur:
        dur = {k: (v / max(scenario_params.base_speed_kmh, 1)) * 60 for k, v in uploaded_dist.items()}
    if uploaded_dur and not uploaded_dist:
        dist = fallback_dist

    if matrix_ok:
        st.success("Đã nhận distance/time matrix upload. Fallback proxy chỉ dùng cho map nếu cần.")
    else:
        st.warning("Chưa có distance/time matrix chính thức. App đang dùng fallback proxy từ địa chỉ + road factor. Đây là giả định mô phỏng, không phải dữ liệu Toyota gốc.")

    if not run_btn:
        st.info("Chọn ngày/scenario rồi bấm **Run simulation**. App sẽ lập tuyến, chạy Monte Carlo và tạo dashboard.")
        with st.expander("Xem nhanh dữ liệu đã đọc"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Suppliers", len(suppliers))
            c2.metric("Demand rows", len(demand))
            c3.metric("Truck types", len(trucks))
            st.dataframe(demand.head(20), use_container_width=True)
        st.stop()

    with st.spinner("Đang tối ưu tuyến và chạy simulation..."):
        baseline_nodes = build_daily_nodes(demand, suppliers, selected_date, trucks, baseline_params)
        scenario_nodes = build_daily_nodes(demand, suppliers, selected_date, trucks, scenario_params)

        base_routes, base_stops, base_events, _ = build_routes_clarke_wright(
            baseline_nodes, trucks, rates, dist, dur, baseline_params
        )
        sc_routes, sc_stops, sc_events, _ = build_routes_clarke_wright(
            scenario_nodes, trucks, rates, dist, dur, scenario_params
        )
        base_route_reps, base_stop_reps, base_mc = run_monte_carlo(base_routes, baseline_nodes, trucks, rates, dist, dur, baseline_params)
        sc_route_reps, sc_stop_reps, sc_mc = run_monte_carlo(sc_routes, scenario_nodes, trucks, rates, dist, dur, scenario_params)
        base_summary = summarize_plan(base_routes, base_mc)
        sc_summary = summarize_plan(sc_routes, sc_mc)
        sc_route_risk = build_route_risk(sc_routes, sc_route_reps)

    # =============================
    # Dashboard
    # =============================
    status, recommendation = decision_recommendation(sc_summary)

    tab_overview, tab_routes, tab_timeline, tab_whatif, tab_data = st.tabs([
        "1. Executive Dashboard", "2. Routes & Map", "3. Simulation Timeline", "4. What-if Comparison", "5. Data & Export"
    ])

    with tab_overview:
        st.subheader("Executive Dashboard")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            kpi_card("Expected total cost", fmt_vnd(sc_summary.get("total_cost_vnd", 0)), f"P95: {fmt_vnd(sc_summary.get('cost_p95_vnd', 0))}", risk_status((sc_summary.get("total_cost_vnd", 0) / max(base_summary.get("total_cost_vnd", 1), 1)) - 1, 0.03, 0.08))
        with c2:
            kpi_card("Truck utilization", fmt_pct(sc_summary.get("utilization", 0)), f"Volume {sc_summary.get('volume_m3', 0):.1f} / Capacity {sc_summary.get('capacity_m3', 0):.1f} m³", risk_status(sc_summary.get("utilization", 0), 0.80, 0.70, higher_is_worse=False))
        with c3:
            kpi_card("Pickup on-time", fmt_pct(sc_summary.get("pickup_on_time_rate", 0)), "Monte Carlo average", risk_status(sc_summary.get("pickup_on_time_rate", 0), 0.95, 0.90, higher_is_worse=False))
        with c4:
            kpi_card("Route late probability", fmt_pct(sc_summary.get("route_late_rate", 0)), "Average route risk", risk_status(sc_summary.get("route_late_rate", 0), 0.10, 0.25))
        with c5:
            kpi_card("CO₂ estimate", f"{sc_summary.get('co2_kg', 0):,.1f} kg", f"{sc_summary.get('distance_km', 0):,.1f} km", "info")

        st.markdown("---")
        color = STATUS_COLORS[status]
        st.markdown(
            f"""
            <div class="decision-box" style="border-left: 8px solid {color};">
                <b>Managerial recommendation:</b><br>{recommendation}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Route risk ranking")
        if not sc_route_risk.empty:
            show = sc_route_risk.copy()
            show["late_probability"] = show["late_probability"].map(lambda x: f"{x:.1%}")
            show["utilization"] = show["utilization"].map(lambda x: f"{x:.1%}")
            show["risk_color"] = sc_route_risk["risk_status"].map({"good": "🟢", "warn": "🟡", "bad": "🔴"})
            st.dataframe(show[["risk_color", "route_id", "sequence", "truck_type", "utilization", "late_probability", "p95_finish_time", "avg_delay_min", "docks"]], use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_cost_waterfall(sc_routes), use_container_width=True)
        with c2:
            if not sc_mc.empty:
                fig = px.histogram(sc_mc, x="pickup_on_time_rate", nbins=20, title="Monte Carlo pickup on-time distribution", color_discrete_sequence=["#1E9E60"])
                fig.update_layout(height=430, xaxis_tickformat=".0%", margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

    with tab_routes:
        st.subheader("Optimized route plan + spatial visualization")
        st.plotly_chart(plot_route_map(sc_routes, suppliers_geo, sc_route_risk), use_container_width=True)
        plan = sc_routes.copy()
        if not plan.empty:
            plan["utilization"] = plan["utilization"].map(lambda x: f"{x:.1%}")
            money_cols = ["transport_cost_vnd", "stop_cost_vnd", "wait_cost_vnd", "overtime_cost_vnd", "total_cost_vnd"]
            for c in money_cols:
                plan[c] = plan[c].map(fmt_vnd)
            st.dataframe(plan[["route_id", "sequence", "truck_type", "volume_m3", "capacity_m3", "utilization", "distance_km", "arrival_tmv_time", "finish_tmv_time", "status", "total_cost_vnd"]], use_container_width=True, hide_index=True)

    with tab_timeline:
        st.subheader("Event-based simulation timeline")
        st.caption("Màu đỏ = delay, vàng = waiting, xanh dương = driving, tím = loading, xanh ngọc = dock.")
        st.plotly_chart(plot_timeline(sc_events, selected_date), use_container_width=True)
        st.markdown("### Stop schedule")
        if not sc_stops.empty:
            stop_show = sc_stops.copy()
            stop_show["late_min"] = stop_show["late_min"].round(1)
            stop_show["wait_min"] = stop_show["wait_min"].round(1)
            stop_show["load_min"] = stop_show["load_min"].round(1)
            st.dataframe(stop_show[["route_id", "supplier_id", "planned_window", "arrival_time", "depart_time", "wait_min", "load_min", "late_min", "volume_m3", "dock", "status"]], use_container_width=True, hide_index=True)

    with tab_whatif:
        st.subheader("Baseline vs Current Scenario")
        comp = pd.DataFrame([
            {"Scenario": "Baseline", **base_summary},
            {"Scenario": "Current scenario", **sc_summary},
        ])
        if not comp.empty:
            display = comp[["Scenario", "total_cost_vnd", "cost_p95_vnd", "routes", "utilization", "pickup_on_time_rate", "route_late_rate", "delay_min", "distance_km", "co2_kg"]].copy()
            display["total_cost_vnd"] = display["total_cost_vnd"].map(fmt_vnd)
            display["cost_p95_vnd"] = display["cost_p95_vnd"].map(fmt_vnd)
            for c in ["utilization", "pickup_on_time_rate", "route_late_rate"]:
                display[c] = display[c].map(lambda x: f"{x:.1%}")
            st.dataframe(display, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_scenario_box(base_mc, sc_mc), use_container_width=True)
        with c2:
            delta_df = pd.DataFrame({
                "KPI": ["Cost", "Utilization", "Pickup on-time", "Route late risk", "CO₂"],
                "Delta": [
                    (sc_summary.get("total_cost_vnd", 0) / max(base_summary.get("total_cost_vnd", 1), 1) - 1),
                    (sc_summary.get("utilization", 0) - base_summary.get("utilization", 0)),
                    (sc_summary.get("pickup_on_time_rate", 0) - base_summary.get("pickup_on_time_rate", 0)),
                    (sc_summary.get("route_late_rate", 0) - base_summary.get("route_late_rate", 0)),
                    (sc_summary.get("co2_kg", 0) / max(base_summary.get("co2_kg", 1), 1) - 1),
                ]
            })
            fig = px.bar(delta_df, x="KPI", y="Delta", text=delta_df["Delta"].map(lambda x: f"{x:+.1%}"), color="Delta", color_continuous_scale=["#1E9E60", "#F2B705", "#D71920"])
            fig.update_layout(height=430, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=30, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab_data:
        st.subheader("Data validation & export")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Demand rows selected date", int((demand["date"] == selected_date).sum()))
        c2.metric("Supplier nodes after split", len(scenario_nodes))
        c3.metric("Truck types available", int((~trucks["truck_type"].isin(unavailable_trucks)).sum()))
        c4.metric("Matrix source", "Uploaded" if matrix_ok else "Fallback proxy")

        with st.expander("Supplier master with proxy coordinates"):
            st.dataframe(suppliers_geo, use_container_width=True, hide_index=True)
        with st.expander("Truck specs"):
            st.dataframe(trucks, use_container_width=True, hide_index=True)
        with st.expander("Rate card"):
            st.dataframe(rates, use_container_width=True, hide_index=True)
        with st.expander("Scenario nodes"):
            st.dataframe(scenario_nodes, use_container_width=True, hide_index=True)

        xlsx = to_excel_download(sc_routes, sc_stops, sc_events, sc_route_risk, sc_summary, scenario_params)
        st.download_button(
            "⬇️ Download result Excel",
            data=xlsx,
            file_name=f"tmv_milkrun_simulation_{pd.Timestamp(selected_date).strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        st.markdown(
            """
            <div class="small-muted">
            Ghi chú: Toyota case data được dùng làm baseline input. Các yếu tố ngẫu nhiên như traffic/loading/dock wait là lớp simulation distribution-based,
            không thay thế dữ liệu gốc. Nếu chưa upload distance/time matrix, app dùng proxy từ địa chỉ để demo decision logic.
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    run_app()
