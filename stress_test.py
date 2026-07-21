"""
stress_test.py  –  production‑ready (2026-07-10)
- Log-transform applies to RTT waiting list, Doctor FTE, and Nurse FTE
- t‑based CI with configurable df floor and sigma_scale
- GARCH code present but inactive
- Fixed missing rmse in compact summary
"""

import sys, os, itertools, warnings, traceback, argparse
from datetime import datetime
import numpy as np, pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.diagnostic import het_arch
from scipy import stats as scistats
from arch import arch_model

warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")
REPORT_DIR = os.path.join(PROJECT_DIR, "reports")
INPUT_FILE = os.path.join(DATA_DIR, "combined_quarterly.csv")
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))
import exog_config as ec

SEASONAL_PERIOD = 4; DIAGNOSTIC_ALPHA = 0.05; INSTABILITY_MARGIN = 1.05
DEFAULT_HORIZONS = [1,2,3,4,8,12]; FAST_HORIZONS = [4,8]; DEFAULT_MIN_TRAIN_SIZE = 6
ORDER_GRID_P = (0,1,2); ORDER_GRID_Q = (0,1,2); ORDER_GRID_D = (0,1)
ORDER_GRID_SEASONAL_P = (0,1); ORDER_GRID_SEASONAL_Q = (0,1); ORDER_GRID_SEASONAL_D = (0,1)
ORDER_SELECTION_TOP_N = 5
BIAS_SUMMARY_LABEL_MATCHES = ["PRODUCTION","ALT","REGRESSION CHECK","BASELINE","EXOG ISOLATION"]
SKIP_METRICS = {"PESA Health spend (level)","GP total appointments (flow)","GP face-to-face appointments (flow)","GP telephone appointments (flow)"}

class Tee:
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.terminal, self.log = sys.stdout, open(filepath,"w",encoding="utf-8")
    def write(self,msg): self.terminal.write(msg); self.log.write(msg)
    def flush(self): self.terminal.flush(); self.log.flush()

def load_series(metric_key):
    df = pd.read_csv(INPUT_FILE); df["quarter"] = pd.to_datetime(df["quarter"])
    df["covid_pulse"] = ((df["quarter"]>="2020-04-01")&(df["quarter"]<="2021-01-01")).astype(float)
    df["post_covid_regime"] = (df["quarter"]>="2021-04-01").astype(float)
    t = np.arange(len(df)); t_centered = t - t.mean(); quadratic = t_centered**2
    df["quadratic_trend"] = quadratic / quadratic.std() if quadratic.std()>0 else quadratic
    df["post_covid_trend_break"] = 0.0; mask = df["quarter"]>="2020-04-01"
    df.loc[mask,"post_covid_trend_break"] = np.arange(1,mask.sum()+1)
    sub = df[df["metric"]==ec.METRIC_NAMES[metric_key]].dropna(subset=["value"]).sort_values("quarter")
    if sub.empty: raise ValueError(f"No data for {metric_key}")
    if metric_key in ec.FIT_START_OVERRIDES:
        cutoff = pd.to_datetime(ec.FIT_START_OVERRIDES[metric_key])
        before = len(sub); sub = sub[sub["quarter"]>=cutoff]
        print(f"  FIT_START_OVERRIDE: {cutoff.date()} ({before}->{len(sub)} obs)")
    return sub.reset_index(drop=True)

def resolve_horizons_and_min_train(metric_key, sub, fast=False):
    cfg = ec.MODEL_CONFIG[metric_key]
    base = FAST_HORIZONS if fast else DEFAULT_HORIZONS
    max_h = cfg.get("horizons", base[-1]); horizons = [h for h in base if h<=max_h] or [max_h]
    n=len(sub); mt=DEFAULT_MIN_TRAIN_SIZE
    while n-mt-max(horizons)<3 and mt>4: mt-=1
    if n-mt-max(horizons)<3: horizons = [h for h in horizons if n-mt-h>=3] or [1]
    if mt!=DEFAULT_MIN_TRAIN_SIZE or horizons!=[h for h in base if h<=max_h]:
        print(f"  Adjusted: MIN_TRAIN={mt}, HORIZONS={horizons}")
    return horizons, mt

def build_candidates(metric_key, fast=False):
    cfg = ec.MODEL_CONFIG[metric_key]; mt = cfg.get("model_type","")
    if mt=="ETS_damped_logit": return [("PRODUCTION ETS damped logit",None,None,None,[])]
    order = tuple(cfg["order"]); seasonal = tuple(cfg["seasonal_order"]); trend = cfg["trend"]
    exog_cols = list(ec.EXOG_CONFIG.get(metric_key,[]))
    p,d,q=order; P,D,Q,s=seasonal
    cand = []; cand.append((f"PRODUCTION {order}x{seasonal} {trend}, exog={exog_cols or 'none'}", order, seasonal, trend, list(exog_cols)))
    alt_trend = "n" if trend=="c" else "c"
    if trend is not None: cand.append((f"ALT: trend flip -> {alt_trend!r}", order, seasonal, alt_trend, list(exog_cols)))
    if fast: return cand
    if metric_key=="RTT % within 18 weeks (performance)":
        cand.append(("AICc GRID WINNER: (0,1,1)x(0,0,0,4) trend=None", (0,1,1),(0,0,0,4),None,[]))
    elif metric_key=="Bed occupancy (level)":
        cand.append(("AICc GRID WINNER: (1,0,0)x(0,0,1,4) trend='c'", (1,0,0),(0,0,1,4),"c",list(exog_cols)))
    if exog_cols:
        cand.append((f"ALT: no exog (drop {exog_cols})", order, seasonal, trend, []))
        if len(exog_cols)>1:
            for col in exog_cols: cand.append((f"EXOG ISOLATION: {col} only", order, seasonal, trend, [col]))
    if D==1: cand.append(("ALT: remove seas diff (D=1->0)", order, (P,0,Q,s), trend, list(exog_cols)))
    elif D==0 and P==0 and Q==0: cand.append(("ALT: add seas MA (D=0,Q=0->Q=1)", order, (0,0,1,s), trend, list(exog_cols)))
    if d==0: cand.append((f"REGRESSION CHECK: d=0->1", (p,1,q), seasonal, "n", list(exog_cols)))
    elif d==1 and D==1: cand.append(("ALT: remove reg diff (d=1->0)", (p,0,q), seasonal, "c", list(exog_cols)))
    if (p,q)!=(1,0): cand.append((f"ALT: AR(1) only ({1},{d},{0})", (1,d,0), seasonal, trend, list(exog_cols)))
    if (p,q)!=(0,1): cand.append((f"ALT: MA(1) only ({0},{d},{1})", (0,d,1), seasonal, trend, list(exog_cols)))
    if (p,q)!=(1,1): cand.append((f"ALT: ARMA(1,1) ({1},{d},{1})", (1,d,1), seasonal, trend, list(exog_cols)))
    cand.append((f"BASELINE: white noise (0,{d},0)", (0,d,0), (0,0,0,s), trend, []))
    seen=set(); dedup=[]
    for c in cand:
        key=(c[1],c[2],c[3],tuple(c[4]))
        if key not in seen: seen.add(key); dedup.append(c)
    return dedup

# ----- Diagnostics (unchanged) -----
def diagnostic_significance(y,exog,cols):
    print("\n--- [1/5] Significance screen (OLS) ---")
    if not cols: print("  No exog columns – skipping."); return
    X=sm.add_constant(exog); model=sm.OLS(y,X).fit()
    for name,coef,pval in zip(cols,model.params[1:],model.pvalues[1:]):
        flag="OK" if pval<DIAGNOSTIC_ALPHA else "FLAG: not significant"
        print(f"  {name:<25} coef={coef:>14.4f}  p={pval:.4f}  [{flag}]")

def diagnostic_quadratic_trend(y):
    print("\n--- [2/5] Quadratic trend check ---")
    t=np.arange(len(y),dtype=float); X=sm.add_constant(np.column_stack([t,t**2]))
    model=sm.OLS(y,X).fit(); coef_t2,pval_t2=model.params[2],model.pvalues[2]
    flag="FLAG: significant curvature" if pval_t2<DIAGNOSTIC_ALPHA else "OK"
    print(f"  t^2 coef={coef_t2:.6f}  p={pval_t2:.4f}  [{flag}]")

def diagnostic_stationarity(y):
    print("\n--- [3/5] Stationarity checks ---")
    def _run(label,series):
        if len(series)<10: print(f"  {label:<20} (too few observations – skipped)"); return
        adf_stat,adf_p,*_=adfuller(series,autolag="AIC")
        kpss_stat,kpss_p,*_=kpss(series,regression="c",nlags="auto")
        adf_verdict="stationary" if adf_p<DIAGNOSTIC_ALPHA else "FLAG: non-stationary (ADF)"
        kpss_verdict="stationary" if kpss_p>=DIAGNOSTIC_ALPHA else "FLAG: non-stationary (KPSS)"
        print(f"  {label:<20} ADF p={adf_p:.4f} [{adf_verdict}]   KPSS p={kpss_p:.4f} [{kpss_verdict}]")
    _run("level",y); _run("1st diff",np.diff(y,n=1))
    if len(y)>SEASONAL_PERIOD: _run(f"seasonal diff ({SEASONAL_PERIOD})",y[SEASONAL_PERIOD:]-y[:-SEASONAL_PERIOD])

def diagnostic_order_selection(y,exog):
    print("\n--- [4/5] Order selection (AICc grid) ---")
    scale=np.nanmax(np.abs(y)); y_scaled=y/scale
    combos=itertools.product(ORDER_GRID_P,ORDER_GRID_D,ORDER_GRID_Q,
                            ORDER_GRID_SEASONAL_P,ORDER_GRID_SEASONAL_D,ORDER_GRID_SEASONAL_Q)
    scored=[]; n_tried,n_failed=0,0
    for p,d,q,P,D,Q in combos:
        order=(p,d,q); seasonal_order=(P,D,Q,SEASONAL_PERIOD); trend="c" if (d==0 and D==0) else None; n_tried+=1
        try:
            model=sm.tsa.statespace.SARIMAX(y_scaled,exog=exog,order=order,seasonal_order=seasonal_order,
                                            trend=trend,enforce_stationarity=True,enforce_invertibility=True)
            res=model.fit(disp=False)
            if np.isfinite(res.aicc): scored.append((res.aicc,order,seasonal_order,trend))
        except: n_failed+=1
    scored.sort(key=lambda r:r[0])
    print(f"  Fitted {n_tried-n_failed}/{n_tried} combinations.")
    for aicc,order,seasonal_order,trend in scored[:ORDER_SELECTION_TOP_N]:
        print(f"  AICc={aicc:>10.2f}  order={order}  seasonal_order={seasonal_order}  trend={trend!r}")
    if not scored: print("  No grid combination converged.")

def diagnostic_instability(sub,candidates):
    print("\n--- [5/5] Instability check ---")
    y_full=sub["value"].values.astype(float); scale=np.nanmax(np.abs(y_full)); y_scaled=y_full/scale
    for label,order,seasonal_order,trend,exog_cols in candidates:
        if order is None: print(f"  {label:<55} [SKIPPED (not SARIMAX)]"); continue
        exog_full=sub[list(exog_cols)].values.astype(float) if exog_cols else None
        try:
            model=sm.tsa.statespace.SARIMAX(y_scaled,exog=exog_full,order=order,seasonal_order=seasonal_order,
                                            trend=trend,enforce_stationarity=True,enforce_invertibility=True)
            res=model.fit(disp=False)
        except Exception as exc: print(f"  {label:<55} FIT FAILED ({type(exc).__name__})"); continue
        ar_roots=np.abs(res.arroots) if len(res.arroots) else np.array([])
        ma_roots=np.abs(res.maroots) if len(res.maroots) else np.array([])
        flags=[]
        if ar_roots.size and ar_roots.min()<INSTABILITY_MARGIN: flags.append(f"near-unit-root AR ({ar_roots.min():.3f})")
        if ma_roots.size and ma_roots.min()<INSTABILITY_MARGIN: flags.append(f"near-non-invertible MA ({ma_roots.min():.3f})")
        print(f"  {label:<55} [{'; '.join(flags) if flags else 'OK'}]")

def run_diagnostic_sequence(sub,cand,dxcols):
    print("\n"+"="*90); print("DIAGNOSTIC SEQUENCE"); print("="*90)
    y_full=sub["value"].values.astype(float)
    exog_full=sub[list(dxcols)].values.astype(float) if dxcols else None
    diagnostic_significance(y_full,exog_full,dxcols); diagnostic_quadratic_trend(y_full)
    diagnostic_stationarity(y_full); diagnostic_order_selection(y_full,exog_full)
    diagnostic_instability(sub,cand)
    print("\nReminder: this report is informational only.")

# ----- Fit & forecast -----
def fit_and_forecast_sarimax(y_scaled, exog, order, seasonal_order, trend, horizon,
                             initialization='approximate_diffuse'):
    model = sm.tsa.statespace.SARIMAX(
        y_scaled, exog=exog, order=order, seasonal_order=seasonal_order,
        trend=trend, enforce_stationarity=True, enforce_invertibility=True,
        initialization=initialization)
    res = model.fit(disp=False, method='lbfgs', maxiter=2000)
    if exog is not None:
        last_val = exog[-1,:]; last_diff = exog[-1,:]-exog[-2,:]
        steps = np.arange(1,horizon+1).reshape(-1,1); future_exog = last_val + steps*last_diff
    else: future_exog = None
    fc = res.get_forecast(steps=horizon, exog=future_exog)
    mean = fc.predicted_mean; mean = mean.to_numpy() if hasattr(mean,"to_numpy") else np.asarray(mean)
    ci = fc.conf_int(alpha=0.05); ci = ci.to_numpy() if hasattr(ci,"to_numpy") else np.asarray(ci)
    return mean[-1], ci[-1,0], ci[-1,1]

def fit_and_forecast_ets(series_prob, horizon):
    eps=1e-6; yc=np.clip(series_prob,eps,1-eps); yl=np.log(yc/(1-yc))
    fit = ExponentialSmoothing(yl, trend='add', damped_trend=True, seasonal=None).fit()
    fc_logit = fit.forecast(steps=horizon); fc_prob = 1/(1+np.exp(-fc_logit))
    resid_std = np.std(fit.resid) if len(fit.resid)>1 else 0.1
    ml = np.asarray(fc_logit); lo = ml-1.96*resid_std; hi = ml+1.96*resid_std
    return fc_prob[-1], (1/(1+np.exp(-lo)))[-1], (1/(1+np.exp(-hi)))[-1]

# ----- t-distribution df estimator -----
def estimate_t_df_from_full_fit(sub, metric_key, cfg):
    y_raw = sub["value"].values.astype(float)
    exog_cols = list(ec.EXOG_CONFIG.get(metric_key,[]))
    apply_log = False
    if (metric_key in ("RTT waiting list (level)", "Doctor FTE (level)", "Nurse FTE (level)")) and cfg.get("transform")=="log":
        if np.all(y_raw>0):
            y = np.log(y_raw)
            apply_log = True
        else:
            y = y_raw
    else:
        y = y_raw
    scale = np.nanmax(np.abs(y)) if not apply_log else 1.0
    y_scaled = y / scale
    exog = sub[exog_cols].values.astype(float) if exog_cols else None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.tsa.statespace.SARIMAX(
                y_scaled, exog=exog,
                order=cfg["order"], seasonal_order=cfg["seasonal_order"],
                trend=cfg["trend"], enforce_stationarity=True, enforce_invertibility=True,
                initialization='approximate_diffuse')
            res = model.fit(disp=False, method='lbfgs', maxiter=2000)
        resid = res.resid
        resid_clean = resid[~np.isnan(resid)]
        if len(resid_clean) < 10:
            return None
        std_resid = (resid_clean - np.mean(resid_clean)) / np.std(resid_clean, ddof=1)
        t_df, t_loc, t_scale = scistats.t.fit(std_resid)
        return t_df
    except Exception:
        return None

# ----- GARCH multiplier function -----
def fit_garch_on_residuals(resid, horizon):
    """Fit GARCH(1,1) on standardized residuals and return per‑horizon variance multipliers."""
    try:
        std_resid = (resid - np.mean(resid)) / np.std(resid, ddof=1)
        garch = arch_model(std_resid, vol='Garch', p=1, q=1, mean='Zero', dist='normal')
        gres = garch.fit(disp='off')
        fcast = gres.forecast(horizon=horizon)
        var_fcast = fcast.variance.values[-1, :]
        unc_var = np.var(std_resid)
        mults = np.sqrt(np.maximum(var_fcast, 0) / unc_var)
        return mults
    except Exception:
        return np.ones(horizon)

# ----- Backtest -----
def run_backtest(sub, candidates, horizons, min_train_size, metric_key):
    y_raw = sub["value"].values.astype(float); n=len(y_raw)
    cfg = ec.MODEL_CONFIG.get(metric_key,{}); mt = cfg.get("model_type","")
    sigma_scale = cfg.get("sigma_scale",1.0)
    print(f"DEBUG sigma_scale = {sigma_scale}")
    ci_method = cfg.get("ci_method", "gaussian")
    t_df = None; t_mult = 1.96; df_use = None
    if ci_method == "t":
        t_df = estimate_t_df_from_full_fit(sub, metric_key, cfg)
        if t_df is not None:
            t_df_floor = cfg.get("t_df_floor", 3.0)
            df_use = max(t_df, t_df_floor)
            t_mult = scistats.t.ppf(0.975, df_use) * np.sqrt((df_use - 2) / df_use)

    # ---- GARCH preparation ----
    garch_multipliers = None
    if cfg.get("garch", False):
        try:
            apply_log = False
            y_full = y_raw.copy()
            if (metric_key in ("RTT waiting list (level)", "Doctor FTE (level)", "Nurse FTE (level)")) and cfg.get("transform")=="log":
                if np.all(y_full > 0):
                    y_full = np.log(y_full)
                    apply_log = True
            scale_full = np.nanmax(np.abs(y_full)) if not apply_log else 1.0
            exog_full = sub[list(ec.EXOG_CONFIG.get(metric_key, []))].values.astype(float) if ec.EXOG_CONFIG.get(metric_key, []) else None
            model_full = sm.tsa.statespace.SARIMAX(y_full / scale_full, exog=exog_full,
                                                   order=cfg["order"], seasonal_order=cfg["seasonal_order"],
                                                   trend=cfg["trend"], enforce_stationarity=True, enforce_invertibility=True,
                                                   initialization='approximate_diffuse')
            res_full = model_full.fit(disp=False, method='lbfgs', maxiter=2000)
            garch_multipliers = fit_garch_on_residuals(res_full.resid, max(horizons))
        except Exception:
            garch_multipliers = None

    apply_log_trans=False; apply_logit_trans=False; scale=1.0; eps=1e-6

    if mt=="ARIMA_logit_mean_reverting":
        apply_logit_trans=True; yc=np.clip(y_raw,eps,1-eps); y_trans=np.log(yc/(1-yc)); scale=1.0
    elif (metric_key in ("RTT waiting list (level)", "Doctor FTE (level)", "Nurse FTE (level)")) and cfg.get("transform")=="log":
        if np.all(y_raw>0):
            apply_log_trans=True; y_trans=np.log(y_raw); scale=1.0
            print("DEBUG: Log transform ACTIVE for", metric_key)
        else:
            y_trans=y_raw; scale=np.nanmax(np.abs(y_trans))
    else:
        y_trans=y_raw; scale=np.nanmax(np.abs(y_trans)); scale=scale if np.isfinite(scale) and scale>0 else 1.0

    ets_c = [c for c in candidates if c[1] is None]; sarimax_c = [c for c in candidates if c[1] is not None]
    exog_cache={}
    for _,order,seas,trend,exog_cols in sarimax_c:
        key=tuple(exog_cols); exog_cache[key]=sub[list(exog_cols)].values.astype(float) if exog_cols else None
    results={label:{h:[] for h in horizons} for label,*_ in candidates}
    for t in range(min_train_size, n-1):
        if sarimax_c: y_train_sarimax = y_trans[:t+1]/scale
        if ets_c: y_train_ets = y_raw[:t+1]
        for h in horizons:
            target = t + h
            if target >= n:
                continue
            for label,order,seas,trend,exog_cols in sarimax_c:
                exog_full=exog_cache[tuple(exog_cols)]; exog_train=exog_full[:t+1] if exog_full is not None else None
                out = fit_and_forecast_sarimax(y_train_sarimax, exog_train, order, seas, trend, h,
                                               initialization='approximate_diffuse')
                if out is None: continue
                fm, lo, hi = out

                se = (hi - lo) / (2.0 * 1.96)

                # t‑based multiplier
                if ci_method == "t" and t_df is not None:
                    halfwidth = t_mult * se
                    lo = fm - halfwidth
                    hi = fm + halfwidth

                # GARCH per‑horizon scaling
                if garch_multipliers is not None and h < len(garch_multipliers):
                    se = se * garch_multipliers[h]
                    lo = fm - 1.96 * se
                    hi = fm + 1.96 * se

                # sigma_scale (can still be used on top, but set to 1.0 for GARCH series)
                if sigma_scale != 1.0:
                    old_halfwidth = (hi - lo) / 2.0
                    centre = (hi + lo) / 2.0
                    halfwidth = old_halfwidth * sigma_scale
                    lo = centre - halfwidth
                    hi = centre + halfwidth
                    if t == min_train_size and h == horizons[0]:
                        print(f"DEBUG sigma applied: old_hw={old_halfwidth:.6f}, new_hw={halfwidth:.6f}, ratio={halfwidth/old_halfwidth}")

                if apply_logit_trans:
                    fm_prob = 1 / (1 + np.exp(-fm))
                    lo_prob = 1 / (1 + np.exp(-lo))
                    hi_prob = 1 / (1 + np.exp(-hi))
                    results[label][h].append({"forecast":fm_prob,"ci_lo":lo_prob,"ci_hi":hi_prob,
                                              "actual":y_raw[target],"last_known":y_raw[t]})
                elif apply_log_trans:
                    results[label][h].append({"forecast":np.exp(fm),"ci_lo":np.exp(lo),"ci_hi":np.exp(hi),
                                              "actual":y_raw[target],"last_known":y_raw[t]})
                else:
                    results[label][h].append({"forecast":fm*scale,"ci_lo":lo*scale,"ci_hi":hi*scale,
                                              "actual":y_raw[target],"last_known":y_raw[t]})
            for label,order,seas,trend,exog_cols in ets_c:
                out = fit_and_forecast_ets(y_train_ets,h)
                if out is None: continue
                fm,lo,hi = out
                if all(np.isfinite([fm,lo,hi])):
                    results[label][h].append({"forecast":fm,"ci_lo":lo,"ci_hi":hi,
                                              "actual":y_raw[target],"last_known":y_raw[t]})
    return results

def compute_metrics(rows):
    if not rows: return None
    f=np.array([r["forecast"] for r in rows]); a=np.array([r["actual"] for r in rows])
    lo=np.array([r["ci_lo"] for r in rows]); hi=np.array([r["ci_hi"] for r in rows])
    lk=np.array([r["last_known"] for r in rows]); err=f-a
    rmse=np.sqrt(np.mean(err**2)); mae=np.mean(np.abs(err)); bias=np.mean(err)
    ad=np.sign(a-lk); fd=np.sign(f-lk); valid=ad!=0
    da=np.mean(ad[valid]==fd[valid]) if valid.sum()>0 else np.nan
    cov=np.mean((a>=lo)&(a<=hi))
    return {"n":len(rows),"rmse":rmse,"mae":mae,"bias":bias,"dir_acc":da,"ci_coverage":cov}

def score_results(results, horizons, candidates):
    print("\n"+"="*90); print("BACKTEST RESULTS"); print("="*90)
    for h in horizons:
        print(f"\n--- Horizon = {h} quarter(s) ---")
        print(f"{'Spec':<55}{'n':>5}{'RMSE':>14}{'MAE':>14}{'Bias':>14}{'DirAcc':>8}{'CICov':>8}")
        for label,*_ in candidates:
            rows=results[label][h]; m=compute_metrics(rows)
            if m is None: print(f"{label:<55}  (no folds)"); continue
            print(f"{label:<55}{m['n']:>5}{m['rmse']:>14.4f}{m['mae']:>14.4f}{m['bias']:>+14.4f}{m['dir_acc']:>8.2f}{m['ci_coverage']:>8.2f}")
    print("\nBias >0 = over-forecasting; <0 = under-forecasting.")
    print("DirAcc = fraction of correct up/down calls.")
    print("CICov  = fraction inside 95% CI (target ~0.95).")

def report_trend_comparison(results, candidates):
    cands = [c for c in candidates if c[1] is not None]
    if len(cands)<2: return
    pairs=[]
    for i in range(len(cands)):
        for j in range(i+1,len(cands)):
            c1,c2=cands[i],cands[j]
            if (c1[1]==c2[1] and c1[2]==c2[2] and tuple(c1[4])==tuple(c2[4]) and c1[3]!=c2[3]): pairs.append((c1,c2))
    if not pairs: return
    for c1,c2 in pairs:
        hcom = sorted(set(results[c1[0]].keys()) & set(results[c2[0]].keys()))
        if not hcom: continue
        hl = max(hcom); m1=compute_metrics(results[c1[0]][hl]); m2=compute_metrics(results[c2[0]][hl])
        if m1 is None or m2 is None: continue
        if m1['ci_coverage']>m2['ci_coverage']: win,los=c1,c2; wm,lm=m1,m2
        elif m2['ci_coverage']>m1['ci_coverage']: win,los=c2,c1; wm,lm=m2,m1
        else:
            if abs(m1['bias'])<=abs(m2['bias']): win,los=c1,c2; wm,lm=m1,m2
            else: win,los=c2,c1; wm,lm=m2,m1
        print(f"\n--- Trend comparison (horizon {hl}) ---")
        print(f"    trend={c1[3]!r}: coverage={m1['ci_coverage']:.2f}, bias={m1['bias']:.1f}")
        print(f"    trend={c2[3]!r}: coverage={m2['ci_coverage']:.2f}, bias={m2['bias']:.1f}")
        print(f"    Selected: trend={win[3]!r} (higher coverage / lower bias)")
        break

def print_bias_summary(results, candidates):
    matched = [label for label,*_ in candidates if any(m in label for m in BIAS_SUMMARY_LABEL_MATCHES)]
    if not matched: return
    hp = sorted({h for label in matched for h in results[label].keys()}); bh = hp[-2:] if len(hp)>=2 else hp
    print("\n"+"="*90); print("BIAS SUMMARY - long horizons"); print(f"(horizons: {bh})"); print("="*90)
    header = f"{'Spec':<55}"+"".join(f"{('h='+str(h)):>28}" for h in bh); print(header)
    for label in matched:
        row = f"{label:<55}"
        for h in bh:
            if h not in results[label]: row+=f"{'not tested':>28}"
            else:
                m=compute_metrics(results[label][h])
                if m is None: row+=f"{'no folds':>28}"
                else:
                    d="OVER" if m['bias']>0 else ("UNDER" if m['bias']<0 else "FLAT")
                    cell = f"{m['bias']:+.4f}({d})"
                    row += f"{cell:>28}"
        print(row)

def residual_diagnostics(sub, metric_key):
    print("\n"+"="*90); print("RESIDUAL DIAGNOSTICS (production spec, full-history fit)"); print("="*90)
    cfg=ec.MODEL_CONFIG[metric_key]; exog_cols=list(ec.EXOG_CONFIG.get(metric_key,[])); mt=cfg.get("model_type","")
    y_raw=sub["value"].values.astype(float)
    if mt=="ARIMA_logit_mean_reverting":
        eps=1e-6; yc=np.clip(y_raw,eps,1-eps); y=np.log(yc/(1-yc)); scale=1.0; y_sc=y/scale; exog=None
        try:
            model=sm.tsa.statespace.SARIMAX(y_sc,exog=exog,order=cfg["order"],seasonal_order=cfg["seasonal_order"],
                                            trend=cfg["trend"],enforce_stationarity=True,enforce_invertibility=True,
                                            initialization='approximate_diffuse')
            res=model.fit(disp=False,method='lbfgs',maxiter=2000)
            resid=res.resid; print(f"  AICc = {res.aicc:.2f}")
        except Exception as exc: print(f"  FIT FAILED: {exc}"); return {"ljung_box":"N/A","jarque_bera":"N/A","arch":"N/A","fit_status":"FIT FAILED"}
        verdict={"ljung_box":"N/A","jarque_bera":"N/A","arch":"N/A","fit_status":"OK"}
    elif mt=="ETS_damped_logit":
        eps=1e-6; yc=np.clip(y_raw,eps,1-eps); yl=np.log(yc/(1-yc))
        fit=ExponentialSmoothing(yl,trend='add',damped_trend=True,seasonal=None).fit()
        resid=fit.resid; n_params=len(fit.params); n=len(yl)
        aicc=fit.aic+2*n_params*(n_params+1)/(n-n_params-1) if n>n_params+1 else np.nan
        print(f"  ETS AICc = {aicc:.2f}"); verdict={"ljung_box":"N/A","jarque_bera":"N/A","arch":"N/A","fit_status":"OK"}
    else:
        apply_log=False
        if (metric_key in ("RTT waiting list (level)", "Doctor FTE (level)", "Nurse FTE (level)")) and cfg.get("transform")=="log":
            if np.all(y_raw>0): apply_log=True; y=np.log(y_raw)
            else: y=y_raw
        else: y=y_raw
        scale=np.nanmax(np.abs(y)) if not apply_log else 1.0; y_sc=y/scale
        exog=sub[exog_cols].values.astype(float) if exog_cols else None
        try:
            model=sm.tsa.statespace.SARIMAX(y_sc,exog=exog,order=cfg["order"],seasonal_order=cfg["seasonal_order"],
                                            trend=cfg["trend"],enforce_stationarity=True,enforce_invertibility=True,
                                            initialization='approximate_diffuse')
            res=model.fit(disp=False,method='lbfgs',maxiter=2000)
            resid=res.resid; print(f"  AICc = {res.aicc:.2f}")
        except Exception as exc: print(f"  FIT FAILED: {exc}"); return {"ljung_box":"N/A","jarque_bera":"N/A","arch":"N/A","fit_status":"FIT FAILED"}
        verdict={"ljung_box":"N/A","jarque_bera":"N/A","arch":"N/A","fit_status":"OK"}

    resid=np.asarray(resid,dtype=float); resid=resid[~np.isnan(resid)]; n=len(resid)
    lb_lags=[lag for lag in (4,8) if lag<n]
    if lb_lags:
        lb=sm.stats.acorr_ljungbox(resid,lags=lb_lags,return_df=True)
        print("\n  Ljung-Box (H0: no autocorrelation left)"); any_flag=False
        for lag,row in lb.iterrows():
            p=row["lb_pvalue"]; flag="OK" if p>=DIAGNOSTIC_ALPHA else "FLAG: residual autocorrelation remains"
            if p<DIAGNOSTIC_ALPHA: any_flag=True
            print(f"    lag={lag:<3} stat={row['lb_stat']:.3f}  p={p:.4f}  [{flag}]")
        verdict["ljung_box"]="FLAG" if any_flag else "OK"
    else: print("\n  Ljung-Box: too few residuals – skipped.")
    if n>=8:
        jb_stat,jb_p,skew,kurt = sm.stats.stattools.jarque_bera(resid)
        flag="OK" if jb_p>=DIAGNOSTIC_ALPHA else "FLAG: residuals non-normal"
        print(f"\n  Jarque-Bera: stat={jb_stat:.3f}  p={jb_p:.4f}  skew={skew:.3f}  kurt={kurt:.3f}  [{flag}]")
        verdict["jarque_bera"]="OK" if jb_p>=DIAGNOSTIC_ALPHA else "FLAG"
    else: print("\n  Jarque-Bera: too few residuals – skipped.")
    if n>=12:
        try:
            nlags=max(1,min(4,n//3)); arch_stat,arch_p,_,_ = het_arch(resid,nlags=nlags)
            flag="OK" if arch_p>=DIAGNOSTIC_ALPHA else "FLAG: heteroskedasticity"
            print(f"\n  ARCH LM: stat={arch_stat:.3f}  p={arch_p:.4f}  [{flag}]")
            verdict["arch"]="OK" if arch_p>=DIAGNOSTIC_ALPHA else "FLAG"
        except Exception: print("\n  ARCH LM: test failed – skipped.")
    else: print("\n  ARCH LM: too few residuals – skipped.")
    print("\n  Note: low power expected with short quarterly series.")
    return verdict

def process_metric(metric_key, fast=False):
    print("\n\n"+"#"*90); print(f"# METRIC: {metric_key}"); print(f"# Series: {ec.METRIC_NAMES[metric_key]}"); print("#"*90)
    sub=load_series(metric_key); print(f"Loaded {len(sub)} obs, {sub['quarter'].iloc[0].date()} to {sub['quarter'].iloc[-1].date()}")
    horizons,mt=resolve_horizons_and_min_train(metric_key,sub,fast); cand=build_candidates(metric_key,fast)
    dxcols=list(ec.EXOG_CONFIG.get(metric_key,[])); print(f"Horizons: {horizons} | Min train: {mt}")
    print(f"Candidates ({len(cand)}): {[c[0] for c in cand]}")
    if not fast: run_diagnostic_sequence(sub,cand,dxcols)
    else: print("\n[FAST MODE] Skipping full diagnostics.")
    res=run_backtest(sub,cand,horizons,mt,metric_key); score_results(res,horizons,cand)
    report_trend_comparison(res,cand)
    if not fast and len(cand)>1: print_bias_summary(res,cand)
    rd=residual_diagnostics(sub,metric_key)
    prod_label=cand[0][0]; lh=max(horizons); pm=compute_metrics(res[prod_label][lh])
    summary=dict(metric_key=metric_key,n_obs=len(sub),longest_horizon_tested=lh,production_spec=prod_label,
                 n_folds=pm['n'] if pm else 0,
                 rmse=pm['rmse'] if pm else np.nan,
                 mae=pm['mae'] if pm else np.nan,
                 bias=pm['bias'] if pm else np.nan, dir_acc=pm['dir_acc'] if pm else np.nan,
                 ci_coverage=pm['ci_coverage'] if pm else np.nan, ljung_box=rd['ljung_box'],
                 jarque_bera=rd['jarque_bera'], arch_heteroskedasticity=rd['arch'], residual_fit_status=rd['fit_status'])
    return summary,res,cand,horizons

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("metric",nargs="?",default=None); parser.add_argument("--list",action="store_true")
    parser.add_argument("--all",action="store_true"); parser.add_argument("--fast",action="store_true")
    args=parser.parse_args()
    if args.list:
        for k in ec.METRIC_NAMES: print(f"  - {k}")
        return
    if args.all: metrics=[k for k in ec.METRIC_NAMES if k not in SKIP_METRICS]
    else:
        if not args.metric: print("ERROR: specify metric or --all"); sys.exit(1)
        if args.metric not in ec.METRIC_NAMES: print(f"Metric '{args.metric}' not found."); sys.exit(1)
        metrics=[args.metric]
    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    for mk in metrics:
        safe=mk.replace(" ","_").replace("/","_").replace("(","").replace(")","")
        log_path=os.path.join(REPORT_DIR,f"backtest_{safe}_log_{timestamp}.txt")
        sum_path=os.path.join(REPORT_DIR,f"backtest_{safe}_summary_{timestamp}.csv")
        sys.stdout=Tee(log_path)
        print("="*90); print("INDIVIDUAL METRIC BACKTEST"); print("="*90)
        print(f"Run started: {datetime.now().isoformat()}\nInput: {INPUT_FILE}\nMetric: {mk}\nFast: {args.fast}")
        try:
            sumrow,_,_,_=process_metric(mk,fast=args.fast)
            print("\n\nCOMPACT SUMMARY"); [print(f"  {k}: {v}") for k,v in sumrow.items()]
            pd.DataFrame([sumrow]).to_csv(sum_path,index=False); print(f"\nSummary saved: {sum_path}")
        except Exception as e: print(f"\n!!! METRIC FAILED: {type(e).__name__}: {e}"); traceback.print_exc()
        print(f"\nLog saved: {log_path}\n"+"="*90)
    print("\nAll backtests completed.")

if __name__=="__main__": main()