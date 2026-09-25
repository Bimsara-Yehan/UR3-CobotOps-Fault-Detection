"""Engineered features shared by training and the serving API, so both compute them identically."""

CURRENT_COLS = [f"Current_J{i}" for i in range(6)]

# delta3 compares a reading with the one LAG_ROWS earlier, so scoring a reading needs
# WINDOW_ROWS consecutive readings: the current one plus LAG_ROWS before it.
LAG_ROWS = 3
WINDOW_ROWS = LAG_ROWS + 1


def add_engineered_features(df, group_col=None):
    """Add total_joint_current and total_joint_current_delta3 to a time-ordered frame of readings.

    delta3 is total_joint_current minus its value LAG_ROWS readings earlier (within `group_col`,
    the cycle id, when given); it is NaN where no such earlier reading exists.
    """
    out = df.copy()
    out["total_joint_current"] = out[CURRENT_COLS].abs().sum(axis=1)
    total = out["total_joint_current"]
    lagged = total.groupby(out[group_col]).shift(LAG_ROWS) if group_col else total.shift(LAG_ROWS)
    out["total_joint_current_delta3"] = total - lagged
    return out
