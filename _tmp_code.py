import subprocess
print(subprocess.run("v4l2-ctl -d /dev/video0 --list-ctrls", shell=True,
                     capture_output=True, text=True).stdout)
for ctrl in ["white_balance_automatic=0", "white_balance_temperature=4000",
             "focus_automatic_continuous=0", "focus_absolute=166",
             "auto_exposure=1", "exposure_time_absolute=250"]:
    r = subprocess.run(f"v4l2-ctl -d /dev/video0 -c {ctrl}", shell=True,
                       capture_output=True, text=True)
    print(f"  set {ctrl:38s} rc={r.returncode} {r.stderr.strip()[:60]}")
print(subprocess.run("v4l2-ctl -d /dev/video0 --get-ctrl=white_balance_automatic,white_balance_temperature,auto_exposure,exposure_time_absolute,focus_automatic_continuous,focus_absolute", shell=True, capture_output=True, text=True).stdout)
