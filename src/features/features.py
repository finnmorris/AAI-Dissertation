import numpy as np
import pandas as pd


# StatsBomb uses a 120 x 80 coordinate system; attacking goal at x=120, y=40.
PITCH_LENGTH = 120
PITCH_WIDTH = 80
GOAL_X = 120
GOAL_CENTRE_Y = 40

# Real goal width 7.32m on a 68m-wide pitch → 7.32/68*80 ≈ 8.61 SB units.
GOAL_WIDTH_SB = 8.61
LEFT_POST_Y = GOAL_CENTRE_Y - GOAL_WIDTH_SB / 2
RIGHT_POST_Y = GOAL_CENTRE_Y + GOAL_WIDTH_SB / 2


def safe_get_nested_value(value, key, default=np.nan):
    """Return value.get(key) for StatsBomb dicts like {"id": 40, "name": "Right Foot"}."""
    if isinstance(value, dict):
        return value.get(key, default)
    return default


def calculate_distance_to_goal(location):
    """Euclidean distance from shot location to centre of goal."""
    if not isinstance(location, list) or len(location) < 2:
        return np.nan
    x, y = location[0], location[1]
    return np.sqrt((GOAL_X - x) ** 2 + (GOAL_CENTRE_Y - y) ** 2)


def calculate_shot_angle(location):
    """Angle (radians) subtended by the goal at the shot location.

    Wider angle = more of the goal is visible from that position.
    Captures information that distance alone misses (e.g. central vs wide).
    """
    if not isinstance(location, list) or len(location) < 2:
        return np.nan

    x, y = location[0], location[1]
    v_left = np.array([GOAL_X - x, LEFT_POST_Y - y])
    v_right = np.array([GOAL_X - x, RIGHT_POST_Y - y])

    norm_left = np.linalg.norm(v_left)
    norm_right = np.linalg.norm(v_right)

    if norm_left == 0 or norm_right == 0:
        return np.nan

    cosine_angle = np.clip(
        np.dot(v_left, v_right) / (norm_left * norm_right), -1.0, 1.0
    )
    return np.arccos(cosine_angle)


def extract_freeze_frame_features(freeze_frame, shot_location):
    """Parse a StatsBomb freeze frame into defender and goalkeeper features.

    freeze_frame: list of player dicts from shot.freeze_frame
    shot_location: [x, y] location of the shot
    """
    features = {
        "num_defenders_in_frame": np.nan,
        "num_teammates_in_frame": np.nan,
        "distance_to_nearest_defender": np.nan,
        "num_defenders_between_shot_and_goal": np.nan,
        "goalkeeper_distance_to_goal": np.nan,
        "goalkeeper_distance_to_shooter": np.nan,
    }

    if not isinstance(freeze_frame, list):
        return features
    if not isinstance(shot_location, list) or len(shot_location) < 2:
        return features

    shot_x, shot_y = shot_location[0], shot_location[1]
    defenders = []
    teammates = []
    goalkeeper_location = None

    for player in freeze_frame:
        if not isinstance(player, dict):
            continue
        loc = player.get("location")
        if not isinstance(loc, list) or len(loc) < 2:
            continue

        if player.get("teammate", False):
            teammates.append(loc)
        else:
            defenders.append(loc)
            if safe_get_nested_value(player.get("position", {}), "name", default="") == "Goalkeeper":
                goalkeeper_location = loc

    features["num_defenders_in_frame"] = len(defenders)
    features["num_teammates_in_frame"] = len(teammates)

    if defenders:
        defender_distances = [
            np.sqrt((d[0] - shot_x) ** 2 + (d[1] - shot_y) ** 2)
            for d in defenders
        ]
        features["distance_to_nearest_defender"] = min(defender_distances)

        defenders_between = 0
        for defender_x, defender_y in defenders:
            between_x = shot_x < defender_x < GOAL_X
            if GOAL_X != shot_x:
                # y-position of the shooter-to-goal-centre line at the defender's x
                expected_y = shot_y + (GOAL_CENTRE_Y - shot_y) * (defender_x - shot_x) / (GOAL_X - shot_x)
                # ±3 SB units is a deliberate approximation of body width in the lane
                close_to_line = abs(defender_y - expected_y) <= 3
            else:
                close_to_line = False

            if between_x and close_to_line:
                defenders_between += 1

        features["num_defenders_between_shot_and_goal"] = defenders_between

    if goalkeeper_location is not None:
        gk_x, gk_y = goalkeeper_location
        features["goalkeeper_distance_to_goal"] = np.sqrt(
            (GOAL_X - gk_x) ** 2 + (GOAL_CENTRE_Y - gk_y) ** 2
        )
        features["goalkeeper_distance_to_shooter"] = np.sqrt(
            (gk_x - shot_x) ** 2 + (gk_y - shot_y) ** 2
        )

    return features


def calculate_previous_event_distance(row):
    """Euclidean distance between the previous event location and the shot location."""
    prev = row["previous_event_location"]
    shot = row["location"]
    if not isinstance(prev, list) or not isinstance(shot, list):
        return np.nan
    if len(prev) < 2 or len(shot) < 2:
        return np.nan
    return np.sqrt((shot[0] - prev[0]) ** 2 + (shot[1] - prev[1]) ** 2)


def get_previous_event_features(events_df):
    """Add previous-event columns to the full events DataFrame before shot filtering.

    Must be called on all events (not just shots) so that the shift correctly
    captures the event immediately before each shot.
    """
    events_df = events_df.sort_values(["match_id", "period", "timestamp"]).copy()
    events_df["previous_event_type"] = events_df.groupby("match_id")["type_name"].shift(1)
    events_df["previous_event_team"] = events_df.groupby("match_id")["team_name"].shift(1)
    events_df["previous_event_location"] = events_df.groupby("match_id")["location"].shift(1)
    events_df["previous_event_same_team"] = (
        events_df["team_name"] == events_df["previous_event_team"]
    ).astype(int)
    return events_df


def create_xg_features(events_df):
    """Convert raw StatsBomb events into a shot-level xG modelling DataFrame.

    Input: DataFrame with columns match_id, type/type_name, team/team_name,
           location, shot, minute, period, and optionally under_pressure.
    Output: one row per shot with engineered features and binary is_goal target.
    """
    df = events_df.copy()

    if "type_name" not in df.columns:
        df["type_name"] = df["type"].apply(
            lambda x: safe_get_nested_value(x, "name") if isinstance(x, dict) else x
        )
    if "team_name" not in df.columns:
        df["team_name"] = df["team"].apply(
            lambda x: safe_get_nested_value(x, "name") if isinstance(x, dict) else x
        )

    # Previous-event features must be derived before filtering to shots.
    df = get_previous_event_features(df)
    shots = df[df["type_name"] == "Shot"].copy()

    # --- Shot metadata ---
    def shot_field(field):
        return shots["shot"].apply(
            lambda x: safe_get_nested_value(x.get(field), "name") if isinstance(x, dict) else np.nan
        )

    shots["shot_outcome"] = shot_field("outcome")
    shots["is_goal"] = (shots["shot_outcome"] == "Goal").astype(int)
    shots["shot_body_part"] = shot_field("body_part")
    shots["shot_type"] = shot_field("type")
    shots["shot_technique"] = shot_field("technique")
    shots["shot_first_time"] = shots["shot"].apply(
        lambda x: int(bool(x.get("first_time", False))) if isinstance(x, dict) else 0
    )
    shots["shot_under_pressure"] = (
        shots["under_pressure"].fillna(False).astype(int)
        if "under_pressure" in shots.columns else 0
    )

    # --- Spatial features ---
    shots["shot_x"] = shots["location"].apply(
        lambda loc: loc[0] if isinstance(loc, list) and len(loc) >= 2 else np.nan
    )
    shots["shot_y"] = shots["location"].apply(
        lambda loc: loc[1] if isinstance(loc, list) and len(loc) >= 2 else np.nan
    )
    shots["distance_to_goal"] = shots["location"].apply(calculate_distance_to_goal)
    shots["shot_angle"] = shots["location"].apply(calculate_shot_angle)
    # Lateral distance from the goal-centre line; smaller = more central = better chance.
    shots["centrality"] = (shots["shot_y"] - GOAL_CENTRE_Y).abs()
    shots["in_penalty_area"] = (
        (shots["shot_x"] >= 102) & (shots["shot_y"] >= 18) & (shots["shot_y"] <= 62)
    ).astype(int)
    shots["in_six_yard_box"] = (
        (shots["shot_x"] >= 114) & (shots["shot_y"] >= 30) & (shots["shot_y"] <= 50)
    ).astype(int)

    # --- Freeze-frame features ---
    shots["shot_freeze_frame"] = shots["shot"].apply(
        lambda x: x.get("freeze_frame") if isinstance(x, dict) else np.nan
    )
    freeze_features_df = pd.DataFrame(
        list(shots.apply(
            lambda row: extract_freeze_frame_features(row["shot_freeze_frame"], row["location"]),
            axis=1,
        )),
        index=shots.index,
    )
    shots = pd.concat([shots, freeze_features_df], axis=1)

    # --- Previous event features ---
    shots["previous_event_was_pass"] = (shots["previous_event_type"] == "Pass").astype(int)
    shots["previous_event_was_carry"] = (shots["previous_event_type"] == "Carry").astype(int)
    shots["previous_event_same_team"] = shots["previous_event_same_team"].fillna(0).astype(int)
    shots["previous_event_distance"] = shots.apply(calculate_previous_event_distance, axis=1)

    # --- Time features ---
    shots["minute"] = pd.to_numeric(shots["minute"], errors="coerce")
    shots["is_late_game"] = (shots["minute"] >= 75).astype(int)
    # Periods 3+ are extra time; these shots may behave differently from open play.
    shots["is_extra_time"] = (shots["period"] > 2).astype(int)

    # Penalties are flagged rather than removed; downstream models should handle
    # them separately given their fixed location and high baseline conversion rate.
    shots["is_penalty"] = (shots["shot_type"] == "Penalty").astype(int)

    modelling_columns = [
        "match_id", "team_name", "minute", "period", "is_goal",
        "shot_x", "shot_y", "distance_to_goal", "shot_angle", "centrality",
        "in_penalty_area", "in_six_yard_box",
        "shot_body_part", "shot_type", "shot_technique",
        "shot_first_time", "shot_under_pressure", "is_penalty",
        "num_defenders_in_frame", "num_teammates_in_frame",
        "distance_to_nearest_defender", "num_defenders_between_shot_and_goal",
        "goalkeeper_distance_to_goal", "goalkeeper_distance_to_shooter",
        "previous_event_type", "previous_event_was_pass", "previous_event_was_carry",
        "previous_event_same_team", "previous_event_distance",
        "is_late_game", "is_extra_time",
    ]
    modelling_columns = [c for c in modelling_columns if c in shots.columns]

    return shots[modelling_columns].copy()
