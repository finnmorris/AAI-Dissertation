import math
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------

# StatsBomb uses a 120 x 80 coordinate system.
# The attacking goal is normally assumed to be at x = 120, y = 40.
PITCH_LENGTH = 120
PITCH_WIDTH = 80
GOAL_X = 120
GOAL_CENTRE_Y = 40

# Approximate goal width converted onto StatsBomb's y-axis scale.
# A real football goal is 7.32m wide. On a 68m-wide pitch, this maps to:
# 7.32 / 68 * 80 = approximately 8.61 StatsBomb coordinate units.
GOAL_WIDTH_SB = 8.61
LEFT_POST_Y = GOAL_CENTRE_Y - GOAL_WIDTH_SB / 2
RIGHT_POST_Y = GOAL_CENTRE_Y + GOAL_WIDTH_SB / 2


# ---------------------------------------------------------------------
# 2. Utility functions
# ---------------------------------------------------------------------

def safe_get_nested_value(value, key, default=np.nan):
    """
    Extracts a value from a nested StatsBomb dictionary.

    StatsBomb event data often stores values as dictionaries, for example:
    shot_body_part = {"id": 40, "name": "Right Foot"}

    This function safely returns the nested value, usually the "name".
    """
    if isinstance(value, dict):
        return value.get(key, default)
    return default


def calculate_distance_to_goal(location):
    """
    Calculates Euclidean distance from the shot location to the centre
    of the goal.

    Reasoning:
    Distance is usually one of the strongest predictors in xG modelling.
    Closer shots are generally much more likely to become goals.
    """
    if not isinstance(location, list) or len(location) < 2:
        return np.nan

    x, y = location[0], location[1]
    return math.sqrt((GOAL_X - x) ** 2 + (GOAL_CENTRE_Y - y) ** 2)


def calculate_shot_angle(location):
    """
    Calculates the angle between the shot location and the two goalposts.

    Reasoning:
    A central shot has a wider visible angle of the goal, while a wide shot
    has a narrower angle. Shot angle captures information that raw distance
    alone misses.
    """
    if not isinstance(location, list) or len(location) < 2:
        return np.nan

    x, y = location[0], location[1]

    # Vectors from shot location to each goalpost
    vector_left_post = np.array([GOAL_X - x, LEFT_POST_Y - y])
    vector_right_post = np.array([GOAL_X - x, RIGHT_POST_Y - y])

    norm_left = np.linalg.norm(vector_left_post)
    norm_right = np.linalg.norm(vector_right_post)

    if norm_left == 0 or norm_right == 0:
        return np.nan

    cosine_angle = np.dot(vector_left_post, vector_right_post) / (norm_left * norm_right)

    # Avoid numerical errors outside [-1, 1]
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    return math.acos(cosine_angle)


def extract_freeze_frame_features(freeze_frame, shot_location):
    """
    Extracts simple defensive and goalkeeper-related features from
    StatsBomb shot freeze-frame data.

    Reasoning:
    StatsBomb freeze frames contain visible player locations at the moment
    of the shot. This can be used to estimate how obstructed or pressured
    the shooter was. StatsBomb has highlighted freeze-frame defender and
    goalkeeper locations as important contextual information for xG modelling.
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

        player_location = player.get("location")
        if not isinstance(player_location, list) or len(player_location) < 2:
            continue

        is_teammate = player.get("teammate", False)
        position = player.get("position", {})

        if is_teammate:
            teammates.append(player_location)
        else:
            defenders.append(player_location)

            # Goalkeeper is normally represented in the position field.
            position_name = safe_get_nested_value(position, "name", default="")
            if position_name == "Goalkeeper":
                goalkeeper_location = player_location

    features["num_defenders_in_frame"] = len(defenders)
    features["num_teammates_in_frame"] = len(teammates)

    if len(defenders) > 0:
        defender_distances = [
            math.sqrt((d[0] - shot_x) ** 2 + (d[1] - shot_y) ** 2)
            for d in defenders
        ]
        features["distance_to_nearest_defender"] = min(defender_distances)

        # Approximate whether defenders are in the shooting lane.
        # This is deliberately simple and interpretable.
        defenders_between = 0

        for defender_x, defender_y in defenders:
            # Defender must be between shooter and goal on x-axis.
            between_x = shot_x < defender_x < GOAL_X

            # Estimate the y-position of the direct line from shooter to goal centre
            # at the defender's x coordinate.
            if GOAL_X != shot_x:
                expected_y_on_line = shot_y + (
                    (GOAL_CENTRE_Y - shot_y) * (defender_x - shot_x) / (GOAL_X - shot_x)
                )
                close_to_shot_line = abs(defender_y - expected_y_on_line) <= 3
            else:
                close_to_shot_line = False

            if between_x and close_to_shot_line:
                defenders_between += 1

        features["num_defenders_between_shot_and_goal"] = defenders_between

    if goalkeeper_location is not None:
        gk_x, gk_y = goalkeeper_location

        features["goalkeeper_distance_to_goal"] = math.sqrt(
            (GOAL_X - gk_x) ** 2 + (GOAL_CENTRE_Y - gk_y) ** 2
        )

        features["goalkeeper_distance_to_shooter"] = math.sqrt(
            (gk_x - shot_x) ** 2 + (gk_y - shot_y) ** 2
        )

    return features


def get_previous_event_features(events_df):
    """
    Adds simple information about the event immediately before each shot.

    Reasoning:
    The action before a shot is often important. Shots following through balls,
    crosses, rebounds or carries may have different scoring probabilities.
    This function creates previous-event context without building a complex
    possession-chain model.
    """
    events_df = events_df.sort_values(["match_id", "period", "timestamp"]).copy()

    events_df["previous_event_type"] = (
        events_df.groupby("match_id")["type_name"].shift(1)
    )

    events_df["previous_event_team"] = (
        events_df.groupby("match_id")["team_name"].shift(1)
    )

    events_df["previous_event_location"] = (
        events_df.groupby("match_id")["location"].shift(1)
    )

    events_df["previous_event_same_team"] = (
        events_df["team_name"] == events_df["previous_event_team"]
    ).astype(int)

    return events_df


# ---------------------------------------------------------------------
# 3. Main feature engineering function
# ---------------------------------------------------------------------

def create_xg_features(events_df):
    """
    Converts raw StatsBomb event data into a shot-level xG modelling dataset.

    Expected input:
    A DataFrame of StatsBomb events containing at least:
    - match_id
    - type_name or type
    - location
    - shot
    - minute
    - period
    - team_name or team

    Output:
    One row per shot, with engineered xG features and binary goal target.
    """

    df = events_df.copy()

    # -----------------------------------------------------------------
    # Standardise common StatsBomb columns
    # -----------------------------------------------------------------

    if "type_name" not in df.columns:
        df["type_name"] = df["type"].apply(
            lambda x: safe_get_nested_value(x, "name") if isinstance(x, dict) else x
        )

    if "team_name" not in df.columns:
        df["team_name"] = df["team"].apply(
            lambda x: safe_get_nested_value(x, "name") if isinstance(x, dict) else x
        )

    # Add previous event information before filtering to shots.
    df = get_previous_event_features(df)

    # -----------------------------------------------------------------
    # Keep only shots
    # -----------------------------------------------------------------

    shots = df[df["type_name"] == "Shot"].copy()

    # -----------------------------------------------------------------
    # Basic shot information
    # -----------------------------------------------------------------

    shots["shot_outcome"] = shots["shot"].apply(
        lambda x: safe_get_nested_value(x.get("outcome"), "name")
        if isinstance(x, dict) else np.nan
    )

    shots["is_goal"] = (shots["shot_outcome"] == "Goal").astype(int)

    shots["shot_body_part"] = shots["shot"].apply(
        lambda x: safe_get_nested_value(x.get("body_part"), "name")
        if isinstance(x, dict) else np.nan
    )

    shots["shot_type"] = shots["shot"].apply(
        lambda x: safe_get_nested_value(x.get("type"), "name")
        if isinstance(x, dict) else np.nan
    )

    shots["shot_technique"] = shots["shot"].apply(
        lambda x: safe_get_nested_value(x.get("technique"), "name")
        if isinstance(x, dict) else np.nan
    )

    shots["shot_first_time"] = shots["shot"].apply(
        lambda x: int(bool(x.get("first_time", False)))
        if isinstance(x, dict) else 0
    )

    shots["shot_under_pressure"] = shots["under_pressure"].fillna(False).astype(int) \
        if "under_pressure" in shots.columns else 0

    # -----------------------------------------------------------------
    # Shot spatial features
    # -----------------------------------------------------------------

    shots["shot_x"] = shots["location"].apply(
        lambda loc: loc[0] if isinstance(loc, list) and len(loc) >= 2 else np.nan
    )

    shots["shot_y"] = shots["location"].apply(
        lambda loc: loc[1] if isinstance(loc, list) and len(loc) >= 2 else np.nan
    )

    shots["distance_to_goal"] = shots["location"].apply(calculate_distance_to_goal)

    shots["shot_angle"] = shots["location"].apply(calculate_shot_angle)

    # Reasoning:
    # Central shots are often higher quality, so this feature measures lateral
    # distance from the centre of the goal.
    shots["centrality"] = abs(shots["shot_y"] - GOAL_CENTRE_Y)

    # Reasoning:
    # Penalty-box location is a simple and interpretable feature that captures
    # whether the shot is from a conventionally dangerous area.
    shots["in_penalty_area"] = (
        (shots["shot_x"] >= 102) &
        (shots["shot_y"] >= 18) &
        (shots["shot_y"] <= 62)
    ).astype(int)

    # Reasoning:
    # Six-yard-box shots are generally very high-probability chances.
    shots["in_six_yard_box"] = (
        (shots["shot_x"] >= 114) &
        (shots["shot_y"] >= 30) &
        (shots["shot_y"] <= 50)
    ).astype(int)

    # -----------------------------------------------------------------
    # Freeze-frame features
    # -----------------------------------------------------------------

    shots["shot_freeze_frame"] = shots["shot"].apply(
        lambda x: x.get("freeze_frame") if isinstance(x, dict) else np.nan
    )

    freeze_features = shots.apply(
        lambda row: extract_freeze_frame_features(
            row["shot_freeze_frame"],
            row["location"]
        ),
        axis=1
    )

    freeze_features_df = pd.DataFrame(list(freeze_features), index=shots.index)
    shots = pd.concat([shots, freeze_features_df], axis=1)

    # -----------------------------------------------------------------
    # Previous event features
    # -----------------------------------------------------------------

    shots["previous_event_was_pass"] = (
        shots["previous_event_type"] == "Pass"
    ).astype(int)

    shots["previous_event_was_carry"] = (
        shots["previous_event_type"] == "Carry"
    ).astype(int)

    shots["previous_event_same_team"] = (
        shots["previous_event_same_team"].fillna(0).astype(int)
    )

    # Previous event distance to shot location
    def previous_event_distance(row):
        previous_location = row["previous_event_location"]
        shot_location = row["location"]

        if not isinstance(previous_location, list) or not isinstance(shot_location, list):
            return np.nan

        if len(previous_location) < 2 or len(shot_location) < 2:
            return np.nan

        return math.sqrt(
            (shot_location[0] - previous_location[0]) ** 2 +
            (shot_location[1] - previous_location[1]) ** 2
        )

    shots["previous_event_distance"] = shots.apply(previous_event_distance, axis=1)

    # -----------------------------------------------------------------
    # Match-time features
    # -----------------------------------------------------------------

    shots["minute"] = pd.to_numeric(shots["minute"], errors="coerce")

    # Reasoning:
    # Late shots may be influenced by fatigue, score state, or tactical urgency.
    shots["is_late_game"] = (shots["minute"] >= 75).astype(int)

    # Reasoning:
    # Extra-time shots may behave differently and can be treated separately.
    shots["is_extra_time"] = (shots["period"] > 2).astype(int)

    # -----------------------------------------------------------------
    # Penalty handling
    # -----------------------------------------------------------------

    # Penalties are usually modelled separately because they are taken from
    # a fixed location and have a much higher baseline conversion probability.
    shots["is_penalty"] = (shots["shot_type"] == "Penalty").astype(int)

    # -----------------------------------------------------------------
    # Select final modelling columns
    # -----------------------------------------------------------------

    modelling_columns = [
        "match_id",
        "team_name",
        "minute",
        "period",
        "is_goal",

        # Spatial features
        "shot_x",
        "shot_y",
        "distance_to_goal",
        "shot_angle",
        "centrality",
        "in_penalty_area",
        "in_six_yard_box",

        # Shot context
        "shot_body_part",
        "shot_type",
        "shot_technique",
        "shot_first_time",
        "shot_under_pressure",
        "is_penalty",

        # Freeze-frame features
        "num_defenders_in_frame",
        "num_teammates_in_frame",
        "distance_to_nearest_defender",
        "num_defenders_between_shot_and_goal",
        "goalkeeper_distance_to_goal",
        "goalkeeper_distance_to_shooter",

        # Previous event features
        "previous_event_type",
        "previous_event_was_pass",
        "previous_event_was_carry",
        "previous_event_same_team",
        "previous_event_distance",

        # Time features
        "is_late_game",
        "is_extra_time",
    ]

    # Keep only columns that exist, in case some metadata varies between files.
    modelling_columns = [col for col in modelling_columns if col in shots.columns]

    xg_df = shots[modelling_columns].copy()

    return xg_df