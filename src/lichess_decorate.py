import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from parquetranger import TableRepo
from tqdm.notebook import tqdm

load_dotenv()

export_dir = Path(os.environ["EXPORT_DIR"])

tournament_sample_path = export_dir / "tournament-sample.parquet"
fixt_trepo = TableRepo(export_dir / "tournament-games", group_cols=["tournament_week"])
ext_trepo = TableRepo(export_dir / "decorated-games", group_cols=["tournament_week"])

uk = "UserId"
tk = "tournamentId"

comm_cols = [tk, "stamp", "termination"]
sides = ["White", "Black"]

streak_bases = ["result_win", "result_lose", "result_draw", "did_berzerk"]
opp_cols = [
    "so_far_termination_abandoned",
    "so_far_termination_normal",
    "so_far_termination_time_forfeit",
    "so_far_result_draw",
    "so_far_result_lose",
    "so_far_result_win",
    "so_far_games",
    "result_win_0_streak",
    "result_win_1_streak",
    "result_lose_0_streak",
    "result_lose_1_streak",
    "result_draw_0_streak",
    "result_draw_1_streak",
    "did_berzerk_0_streak",
    "did_berzerk_1_streak",
    "base_points",
    "didnt_lose",
    "did_berzerk",
    "berzerked_first",
    "on_streak",
    "points_won",
    "current_points",
    "performance_rating",
    "current_position_based_on_points",
    "current_position_based_on_points_and_pr",
    "initial_position_based_on_elo",
    "so_far_termination_rules_infraction",
    "current_rank_rate",
    "initial_rank_rate",
    "overachievement_rate",
]


# https://en.wikipedia.org/wiki/Elo_rating_system#Performance_rating
PS_K = 400


def get_gdf_base(gdf: pd.DataFrame):
    return gdf.assign(
        games_so_far=lambda df: df.assign(c=1).groupby(uk)["c"].transform("cumsum"),
        berzerked_first=lambda df: df.groupby(uk)["did_berzerk"].transform("first"),
        on_streak=lambda df: df.groupby(uk)["did_win"]
        .rolling(2)
        .sum()
        .groupby(uk)
        .shift(1)
        .reset_index(level=0, drop=True)
        >= 2,
        points_won=lambda df: df["base_points"] * (1 + df["on_streak"])
        + (df["did_win"] * df["did_berzerk"]),
        current_points=lambda df: df.groupby(uk)["points_won"].transform("cumsum"),
        performance_rating_addition=lambda df: df["OppElo"]
        + PS_K * np.where(df["did_win"], 1, np.where(df["result"] == "lose", -1, 0)),
        performance_rating=lambda df: df.groupby(uk)[
            "performance_rating_addition"
        ].transform("cumsum")
        / df["games_so_far"],
        points_and_pr=lambda df: df[["current_points", "performance_rating"]].apply(
            tuple, axis=1
        ),
        current_position_based_on_points=lambda df: [
            df.iloc[:i, :]
            .drop_duplicates(uk, keep="last")["current_points"]
            .pipe(
                lambda s: np.searchsorted(
                    (-s).sort_values(), -df.iloc[i - 1, :]["current_points"]
                )
                + 1
            )
            for i in range(1, df.shape[0] + 1)
        ],
        current_position_based_on_points_and_pr=lambda df: [
            df.iloc[:i, :]
            .drop_duplicates(uk, keep="last")["points_and_pr"]
            .rank(ascending=False)
            .iloc[-1]
            for i in range(1, df.shape[0] + 1)
        ],
        initial_position_based_on_elo=lambda df: df.groupby(uk)["Elo"]
        .first()
        .rank()
        .reindex(df[uk])
        .values,
    )


class RankedExporter:
    def __init__(self, games_df: pd.DataFrame):

        self.tournament_sample_df = pd.read_parquet(tournament_sample_path)

        self.fixtures_df = self.get_fixtures(games_df)
        self.ranked_df = pd.concat(
            map(
                lambda gtup: get_gdf_base(gtup[1]),
                tqdm(self.fixtures_df.sort_values("stamp").groupby("tournamentId")),
            )
        ).pipe(lambda df: df.set_index(df.index.rename("fixture_id")))

        self.dummy_basis = pd.get_dummies(
            self.ranked_df.set_index([uk, tk], append=True)[
                ["termination", "result", "did_berzerk", "didnt_lose"]
            ]
        ).astype(int)

        self.full_streak_df = pd.concat(map(self.get_streak_df, streak_bases), axis=1)

        extended_ranked_df = self.ranked_df.join(
            self.dummy_basis.assign(games=1)
            .groupby([tk, uk])
            .cumsum()
            .pipe(lambda db: db - self.dummy_basis.assign(games=1))
            .reset_index([uk, tk], drop=True)
            .rename(columns=lambda s: f"so_far_{s}")
        ).join(self.full_streak_df.reset_index([uk, tk], drop=True))

        streak_cols = [
            c
            for c in extended_ranked_df.columns
            if c.endswith("_streak") and c != "on_streak"
        ]

        fixed_streaks = extended_ranked_df.pipe(
            lambda df: df.sort_values("stamp").groupby([tk, uk])[streak_cols].shift(1)
        )

        extended_ranked_df = extended_ranked_df.drop(streak_cols, axis=1).join(
            fixed_streaks.fillna(0)
        )
        ext_trepo.extend(self.get_decorated_fixtures(extended_ranked_df))

    def get_streak_df(self, dumk):
        _shifted = self.dummy_basis.groupby([tk, uk], as_index=False)[dumk].shift()
        _streak_id = (self.dummy_basis[dumk] != _shifted).groupby([tk, uk]).cumsum()
        _streak_len = (
            _streak_id.to_frame().assign(c=1).groupby([tk, uk, dumk])["c"].cumsum()
        )

        return (
            pd.concat([self.dummy_basis[dumk], _streak_len], axis=1)
            .pivot_table(index=_shifted.index.names, columns=dumk, values="c")
            .rename(columns=lambda s: f"{dumk}_{s}_streak")
            .fillna(0)
        )

    def get_decorated_fixtures(self, extended_ranked_df):
        return (
            extended_ranked_df.merge(
                self.tournament_sample_df.loc[
                    :,
                    [
                        "fullName",
                        "startsAt",
                        "finishesAt",
                        "nbPlayers",
                        "tournament_week",
                    ],
                ],
                left_on="tournamentId",
                right_index=True,
            )
            .assign(
                stage_of_tournament=lambda df: (
                    df["stamp"].astype(int) / 1e6 - df["startsAt"]
                )
                / (df["finishesAt"] - df["startsAt"]),
                elo_diff=lambda df: df["Elo"] - df["OppElo"],
                elo_diff_rate=lambda df: df["elo_diff"] / df["Elo"],
                current_rank_rate=lambda df: df[
                    "current_position_based_on_points_and_pr"
                ]
                / df["nbPlayers"],
                initial_rank_rate=lambda df: df["initial_position_based_on_elo"]
                / df["nbPlayers"],
                overachievement_rate=lambda df: df["initial_rank_rate"]
                - df["current_rank_rate"],
            )
            .pipe(
                lambda df: df.merge(
                    df.reindex(opp_cols + ["stamp", uk], axis=1).rename(
                        columns={k: f"opposition_{k}" for k in opp_cols} | {uk: "OppId"}
                    )
                )
            )
            .drop_duplicates(subset=["stamp", uk])
        )

    def get_fixtures(self, games_df: pd.DataFrame):
        return (
            games_df.assign(
                stamp=lambda df: pd.to_datetime(df["UTCDate"] + " " + df["UTCTime"]),
                termination=lambda df: df["Termination"]
                .str.replace(" ", "_")
                .str.lower(),
            )
            .pipe(
                lambda df: pd.concat(
                    df.loc[:, comm_cols].assign(
                        UserId=df[side],
                        OppId=df[opp],
                        Elo=df[f"{side}Elo"].astype(float),
                        OppElo=df[f"{opp}Elo"].astype(float),
                        Result=df["Result"]
                        if side == sides[0]
                        else df["Result"].str[::-1],
                        result=lambda df: df["Result"].pipe(
                            lambda s: np.where(
                                s == "1-0", "win", np.where(s == "0-1", "lose", "draw")
                            )
                        ),
                        startTime=pd.to_timedelta(df[f"{side}Start"]).dt.seconds,
                    )
                    for side, opp in zip(sides, sides[::-1])
                )
            )
            .reset_index(drop=True)
            .assign(
                base_points=lambda df: df["Result"].pipe(
                    lambda s: np.where(s == "1-0", 2, np.where(s == "0-1", 0, 1))
                ),
                didnt_lose=lambda df: df["base_points"] != 0,
                did_win=lambda df: df["base_points"] == 2,
                did_berzerk=lambda df: df["startTime"]
                < self.tournament_sample_df["clock__limit"]
                .reindex(df["tournamentId"])
                .values,
            )
        )


def proc_drop(df):
    RankedExporter(df)
