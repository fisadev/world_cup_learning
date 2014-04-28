# coding: utf-8
"""
Functions that return the data in the files, sometimes raw, others with some
cleaning or summarization.
"""
import pandas as pd


RAW_MATCHES_FILE = 'raw_matches.csv'
RAW_WINNERS_FILE = 'raw_winners.csv'
TEAM_RENAMES_FILE = 'team_renames.csv'


def apply_renames(column):
    """Apply team renames to a team column from a dataframe."""
    with open(TEAM_RENAMES_FILE) as renames_file:
        renames = dict(l.strip().split(',')
                       for l in renames_file.readlines()
                       if l.strip())

        def renamer(team):
            return renames.get(team, team)

    return column.apply(renamer)


def get_matches():
    """Create a dataframe with matches info."""
    matches = pd.DataFrame.from_csv(RAW_MATCHES_FILE)
    for column in ('team1', 'team2'):
        matches[column] = apply_renames(matches[column])

    return matches


def get_winners():
    """Create a dataframe with podium positions info."""
    winners = pd.DataFrame.from_csv(RAW_WINNERS_FILE)
    winners.team = apply_renames(winners.team)

    return winners


def get_team_stats():
    """Create a dataframe with useful stats for each team."""
    winners = get_winners()
    matches = get_matches()

    teams = set(matches.team1.unique()).union(matches.team2.unique())

    stats = pd.DataFrame(list(teams), columns=['team'])
    stats['matches_played'] = 0
    stats['matches_won'] = 0
    stats['matches_won_percent'] = 0

    stats['years_played'] = 0

    stats['podium_score'] = 0
    stats['podium_score_yearly'] = 0

    stats['cups_won'] = 0
    stats['cups_won_yearly'] = 0

    stats = stats.set_index('team')

    for team in teams:
        their_matches = matches[(matches.team1 == team) |
                                (matches.team2 == team)]
        stats.loc[team, 'matches_played'] = len(their_matches)

        # wins where the team was on the left side (team1)
        wins1 = their_matches[(their_matches.team1 == team) &
                              (their_matches.score1 > their_matches.score2)]
        # wins where the team was on the right side (team2)
        wins2 = their_matches[(their_matches.team2 == team) &
                              (their_matches.score2 > their_matches.score1)]

        stats.loc[team, 'matches_won'] = len(wins1) + len(wins2)

        stats.loc[team, 'years_played'] = len(their_matches.year.unique())

        their_podiums = winners[winners.team == team]
        to_score = lambda position: 5 - position  # better position = more score
        stats.loc[team, 'podium_score'] = their_podiums.position.map(to_score).sum()

        stats.loc[team, 'cups_won'] = len(their_podiums[their_podiums.position == 1])

    stats['matches_won_percent'] = stats['matches_won'] / stats['matches_played'] * 100.0
    stats['podium_score_yearly'] = stats['podium_score'] / stats['years_played']
    stats['cups_won_yearly'] = stats['cups_won'] / stats['years_played']

    return stats
