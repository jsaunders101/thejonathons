import json, tempfile, shutil, sys
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg")
sys.path.insert(0, "/Users/scottseneca/repos/CSHL_M1_Arousal")
from whisk_bouts import condition
import tifffile
TMP = Path(tempfile.mkdtemp(prefix="fc_")); BD = TMP/"bundles"; BD.mkdir(parents=True)
rng = np.random.default_rng(7); TRUE_LAG_S = 0.5
SPEC = [("thormouse1_spont",30.0,119,28777,28978,36000,24,
         ["laser_trigger","whisker_pad","paw","paw_at_nose","eye","wheel"]),
        ("thormouse2_spont",15.0,61,13960,14647,27000,34,
         ["laser_trigger","whisker_pad","paw","wheel"])]
for name,cam,on,off,n_cam,n2_full,nroi,boxes in SPEC:
    me = np.abs(rng.normal(8,3,(len(boxes),n_cam))); wh = np.zeros(n_cam)
    for k in rng.choice(n_cam,60,replace=False): wh[max(0,k-int(cam)):k+int(cam)] += 30
    me[boxes.index("whisker_pad")] += wh; me[:,0] = 0.0
    lum = rng.normal(345,3,(len(boxes),n_cam)); lum[0,:on]=205; lum[0,off+1:]=205
    proc = TMP/name/"proc"; proc.mkdir(parents=True)
    np.savez_compressed(proc/f"{name}_boxtraces.npz", motion_energy=me.astype(np.float32),
        traces=lum.astype(np.float32), box_names=np.array(boxes),
        box_coords=np.zeros((len(boxes),4),int), trim_tail=3, truncated=False)
    d = np.load(proc/f"{name}_boxtraces.npz")
    me = d["motion_energy"].astype(float); lum = d["traces"].astype(float)
    P,f2p = 1/30.0,30.0; cam_dt = 1.0/cam
    t_full = np.arange(n2_full)*P; cov = (off-on)*cam_dt
    T = int(np.searchsorted(t_full,cov,side="right")); t = t_full[:T]
    t_cam = (np.arange(n_cam)-on)*cam_dt
    mer = me.copy(); mer[:,0]=mer[:,1]; mer[:,on]=mer[:,on+1]
    beh_me = np.vstack([np.interp(t,t_cam,r) for r in mer])
    beh_lum = np.vstack([np.interp(t,t_cam,r) for r in lum])
    beh_env = np.vstack([np.interp(t,t_cam,condition(r,cam,0.33)) for r in mer])
    drive = beh_env[boxes.index("whisker_pad")]; drive=(drive-drive.mean())/drive.std()
    sh = int(round(TRUE_LAG_S*f2p)); dl = np.r_[np.zeros(sh),drive[:-sh]]
    dff = rng.normal(0,.35,(nroi,T)); dff[:nroi//2] += 0.9*dl
    ca = TMP/f"ca_{name}"; ca.mkdir()
    for fn,arr in (("dff.npy",dff),("traces_raw.npy",dff+5),("F0.npy",np.full(nroi,5.)),
        ("roi_npix.npy",np.full(nroi,42)),("mean_img.npy",rng.random((256,256))),
        ("corr_map.npy",rng.random((256,256))),("shifts_yx.npy",np.zeros((2,n2_full)))):
        np.save(ca/fn,arr)
    (ca/"_COMPLETE").write_text("")
    (ca/"params.json").write_text(json.dumps({"fps":30.0,"n_frames":n2_full}))
    tifffile.imwrite(ca/"roi_map.tif", rng.integers(0,nroi+1,(256,256)).astype(np.uint16))
    art = (t<cam_dt)|(t>t[-1]-cam_dt)
    M = np.vstack([dff,beh_me,beh_lum,beh_env])
    Mn = [f"roi_{i:04d}" for i in range(nroi)]+[f"me_{b}" for b in boxes]+ \
         [f"lum_{b}" for b in boxes]+[f"env_{b}" for b in boxes]
    meta = dict(run=name,on=on,off=off,tseries_fps=30.0,tseries_frame_period_s=P,
        tseries_duration_s=n2_full/f2p,tseries_clock="constant_rate",cam_fps_used=cam,
        interp_factor=cam_dt/P,coverage_end_s=cov,n_2p_full=n2_full,n_2p_dropped=n2_full-T,
        n_cam=n_cam,n_2p=T,n_roi=nroi,n_box=len(boxes),box_names=boxes,
        origin_uncertainty_s=cam_dt,map_mode="single_anchor_known_fps",
        beh_npz=str(proc/f"{name}_boxtraces.npz"),beh_vintage="aug9+",beh_truncated=False,
        beh_trim_tail=3,ca_folder=str(ca),ca_layout="v6_npy",conditioned=True,env_s=0.33)
    with open(BD/f"{name}_aligned.npz","wb") as fh:
        np.savez_compressed(fh,meta=json.dumps(meta),t=t,dff=dff,beh_me=beh_me,
            beh_lum=beh_lum,beh_env=beh_env,beh_names=np.array(boxes),artifact=art,M=M,
            M_names=np.array(Mn),F_raw=dff+5,roi_npix=np.full(nroi,42),
            roi_xy=rng.random((nroi,2))*256)
NB = Path("/Users/scottseneca/repos/CSHL_M1_Arousal/handoff/final_correlation.ipynb")
SRC = ["".join(c["source"]) for c in json.loads(NB.read_text())["cells"] if c["cell_type"]=="code"]
ns = {"__name__":"__main__"}
exec(f'from pathlib import Path\nBUNDLE_DIR=Path(r"{BD}")\nFIG_DIR=Path(r"{TMP}/fig")\n'
     f'RUN_NAMES={[s[0] for s in SPEC]!r}\nPRIMARY_BOX="whisker_pad"\nENV_S=0.33\n'
     f'MAX_LAG_S=10.0\nNULL_MIN_S=15.0\nREPO_DIR=r"/Users/scottseneca/repos/CSHL_M1_Arousal"\n', ns)
import io, contextlib
for i,src in enumerate(SRC[1:],1):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf): exec(src, ns)
    except SystemExit: pass
    except Exception as e:
        import traceback; print(f"*** CELL {i} FAILED: {type(e).__name__}: {e}"); traceback.print_exc(); break
    out = buf.getvalue()
    if "STAGE 1b" in out or "n_eff" in out or "|r|" in out or "common to all" in out \
       or "cleared for analysis" in out or "wrote" in out or "env <-" in out:
        print(out.rstrip())
print(f"\nTRUE planted lag = {TRUE_LAG_S:+.2f}s (behaviour leads)")
shutil.rmtree(TMP, ignore_errors=True)
