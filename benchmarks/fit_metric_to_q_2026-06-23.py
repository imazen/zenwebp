import pyarrow.parquet as pq, glob, collections, numpy as np
D = "/tmp/claude-1000/-home-lilith-work-zen/1af79882-d5a9-4e7b-ba8f-793eea6cce30/scratchpad/omni"
cols = ['codec', 'q', 'score_ssim2_gpu', 'score_butteraugli_max_gpu']
recs = collections.defaultdict(list); allp = []
for fp in glob.glob(D + "/**/*.parquet", recursive=True):
    try: t = pq.read_table(fp, columns=cols)
    except Exception: continue
    for c, q, s, b in zip(t['codec'].to_pylist(), t['q'].to_pylist(),
                          t['score_ssim2_gpu'].to_pylist(), t['score_butteraugli_max_gpu'].to_pylist()):
        if s is None or b is None or q is None: continue
        recs[c].append((q, s, b)); allp.append((s, b))
allp = np.array(allp)

def pava_decreasing(y, w):
    # Pool-adjacent-violators for a NON-increasing fit.
    y = list(map(float, y)); w = list(map(float, w))
    lvl = [[yi, wi] for yi, wi in zip(y, w)]  # [mean, weight]
    i = 0
    while i < len(lvl) - 1:
        if lvl[i][0] < lvl[i+1][0] - 1e-12:  # violation: should be >=
            tw = lvl[i][1] + lvl[i+1][1]
            lvl[i] = [(lvl[i][0]*lvl[i][1] + lvl[i+1][0]*lvl[i+1][1]) / tw, tw]
            del lvl[i+1]
            if i > 0: i -= 1
        else:
            i += 1
    out = []
    for m, w_ in lvl:
        out += [m] * int(round(w_ / min(w)))  # not exact, rebuild below
    # rebuild by replaying merges against original counts
    res = []; idx = 0
    for m, w_ in lvl:
        n = 0; acc = 0.0
        while idx < len(y) and acc < w_ - 1e-9:
            acc += w[idx]; idx += 1; n += 1
        res += [m] * n
    return np.array(res)

def forward(codec):
    a = np.array(recs[codec]); qs = sorted(set(a[:,0].tolist()))
    s_med = []; b_med = []; wts = []
    for q in qs:
        m = a[:,0] == q
        s_med.append(float(np.median(a[m,1]))); b_med.append(float(np.median(a[m,2]))); wts.append(int(m.sum()))
    qs = np.array(qs, float); wts = np.array(wts, float)
    # monotone: ssim2 increasing in q -> fit non-decreasing = -pava(-x)
    s_mono = -pava_decreasing(-np.array(s_med), wts)
    # butter-max decreasing in q
    b_mono = pava_decreasing(np.array(b_med), wts)
    return qs, s_mono, b_mono, wts

def invert_to_q(metric_mono, qs, ascending):
    # build (metric, q) sorted ascending in metric, strictly monotone
    pairs = sorted(zip(metric_mono.tolist(), qs.tolist()))
    out = []
    for mv, qv in pairs:
        if out and mv <= out[-1][0] + 1e-6:   # dedup flats: keep higher quality (safer)
            if (ascending and qv > out[-1][1]) or ((not ascending) and qv < out[-1][1]):
                out[-1] = (out[-1][0], qv)
            continue
        out.append((mv, qv))
    return out

def rust(name, pairs, fa, fb):
    body = ",\n            ".join(f"({fa.format(a)}, {fb.format(b)})" for a,b in pairs)
    print(f"        const {name}: &[(f32, f32)] = &[\n            {body},\n        ];")

for codec in ('zenwebp','zenjpeg'):
    qs, s, b, w = forward(codec)
    print(f"\n// ===== {codec} (monotonized; n/q range {int(w.min())}..{int(w.max())}) =====")
    bm = invert_to_q(b, qs, ascending=False)   # bmax ascending -> q descending
    sm = invert_to_q(s, qs, ascending=True)    # ssim2 ascending -> q ascending
    rust("BUTTER_MAX_TO_Q", bm, "{:.2f}", "{:.0f}")
    rust("SSIM2_TO_Q", sm, "{:.1f}", "{:.0f}")

# universal ssim2 -> butter-max (pooled), monotone in ssim2 (decreasing)
print("\n// ===== universal ssim2 -> butteraugli-max (pooled, for zenjxl) =====")
centers = list(range(95, 29, -5)); uni = []
for c in centers:
    m = np.abs(allp[:,0]-c) <= 2.0
    if m.sum() >= 50: uni.append((float(c), float(np.median(allp[m,1]))))
# ssim2 descending here; emit ascending in ssim2 for interp, b decreasing as ssim2 rises
uni_sorted = sorted(uni)  # ascending ssim2
# enforce b strictly decreasing as ssim2 increases
clean = []
for s_, b_ in uni_sorted:
    if clean and b_ >= clean[-1][1]: b_ = clean[-1][1] - 0.01
    clean.append((s_, b_))
rust("SSIM2_TO_BUTTER_MAX", clean, "{:.0f}", "{:.2f}")
