# coding=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns


CONST_RATING = 1500.


def proba_elo(rating1, rating2, norm=400.):
    """
    >>> proba_elo(2800, 2750)[0]
    0.5714631174083814
    >>> proba_elo(2800, 2500)[0]
    0.8490204427886767
    """
    diff = float(rating2 - rating1) / norm
    win_exp1 = 1. / (1. + 10. ** diff)
    return win_exp1, 1. - win_exp1


def diff_needed_to_proba(proba, norm=400.):
    return - np.log(1. / proba - 1.) / np.log(10.) * norm


def updated_elo(rating1, rating2, result1, k=15., norm=400.):
    result2 = 1. - result1
    win_exp1, win_exp2 = proba_elo(rating1, rating2, norm)
    upd = k * (result1 - win_exp1)
    new_rating1 = rating1 + upd
    new_rating2 = rating2 - upd
    return (win_exp1, win_exp2), new_rating1, new_rating2, upd


def h2h_match_result(score1, score2):
    """
    >>> h2h_match_result(5, 1) == 5. / 6
    True
    >>> h2h_match_result(0, 1) == 2.5 / 6
    True
    >>> h2h_match_result(3, 1) == 4. / 6
    True
    """
    draws_add = (6. - (score1 + score2)) / 2
    return float(score1 + draws_add) / 6


def sorted_rating(ratings, n_best=None):
    if n_best is None:
        n_best = len(ratings)
    return [(i + 1, t, r) for i, (t, r) in enumerate(sorted(ratings.items(), key=lambda x: -x[1])[:n_best])]


def print_ratings(ratings, n_best=None):
    for place, t, r in sorted_rating(ratings, n_best):
        print place, t, r


def sim_season(match_log, start_ratings, teams, k, norm, mse_cold_start=0, with_series=True):
    ratings = start_ratings.copy()
    square_errors = []
    if with_series:
        ratings_series = {t: [start_ratings[t]] for t in teams}
        dates = {t: [match_results['date'].values[0]] for t in teams}
        updates = {t: [0] for t in teams}
        tournaments = {t: ['pre'] for t in teams}
    for i, (ind, row) in enumerate(match_log.iterrows()):
        result1 = h2h_match_result(row['score1'], row['score2'])
        team1 = row['team1']
        team2 = row['team2']
        win_p, ratings[team1], ratings[team2], upd = updated_elo(ratings[team1], ratings[team2], result1, k, norm)
        if i > mse_cold_start:
            square_errors.append((result1 - win_p[0]) ** 2)
        if with_series:
            ratings_series[team1].append(ratings[team1])
            ratings_series[team2].append(ratings[team2])
            dates[team1].append(row['date'])
            dates[team2].append(row['date'])
            updates[team1].append(upd)
            updates[team2].append(-upd)
            tournaments[team1].append(row['tournament'])
            tournaments[team2].append(row['tournament'])
    if with_series:
        return ratings, np.mean(square_errors), (ratings_series, dates, updates, tournaments)
    else:
        return ratings, np.mean(square_errors)


def validate_params(start_ratings, k_list, norm_list):
    min_error = 1.
    min_k = 0
    min_norm = 0
    for k_ in k_list:
        for norm_ in norm_list:
            new_ratings, error = sim_season(match_results, start_ratings, all_teams,
                                            k_, norm_, mse_cold_start=len(match_results) / 2, with_series=False)
            if error < min_error:
                min_error = error
                min_k = k_
                min_norm = norm_
                print k_, norm_, error
    return min_k, min_norm, min_error


def plot_series(series, dates, teams, highlight=None):
    all_dates = set()
    for t, d in dates.items():
        all_dates = all_dates.union(d)
    series_df = pd.DataFrame()
    series_df['date'] = np.asarray(sorted(list(all_dates)))
    for t in teams:
        df = pd.DataFrame()
        df[t] = np.asarray(series[t])
        df['date'] = np.asarray(dates[t])
        df = df.drop_duplicates('date', keep='last')
        df_missing = pd.DataFrame()
        df_missing['date'] = list(all_dates.difference(df.date.values))
        df = df.append(df_missing, ignore_index=True)
        df = df.sort_values('date')
        df = df.fillna(method='ffill')
        series_df = series_df.merge(df, on='date')

    if highlight is not None:
        cmap = ['#95a5a6'] * len(teams)
        cmap[teams.index(highlight)] = '#e74c3c'
        legend = False
    else:
        cmap = sns.color_palette('Spectral', len(teams)).as_hex()
        legend = True

    series_df.plot(x='date', colormap=ListedColormap(cmap), figsize=(15, 8), legend=legend)
    if highlight is not None:
        plt.title(highlight.decode('utf-8'))


path = ''
match_results = pd.read_csv(path + 'h2h-all-matches.csv')
match_results = match_results.sort_values('date', ascending=True)
all_teams = set(match_results['team1'].unique())
all_teams.update(match_results['team2'].unique())
const_ratings = {t: CONST_RATING for t in all_teams}

# Validate params for const rating 1500
_k, best_norm, e = validate_params(const_ratings, np.linspace(10, 30, 25).tolist(), np.linspace(300, 700, 25).tolist())
print 'Best norm', best_norm, 'gives error', e  # 0.04685
# best_norm = 450.

# Choose additional rating for higher divisions
add_rating = diff_needed_to_proba(0.75, best_norm) / 7
print 'Additional rating by PL division', add_rating

# Reinitialize ratings
initial_ratings = {t: CONST_RATING - (add_rating * 2.36585365854) for t in all_teams}

for t in match_results['tournament'].unique():
    tourn_teams = set(match_results[match_results.tournament == t]['team1'].unique()).union(
        match_results[match_results.tournament == t]['team2'].unique())
    if t in ['РФПЛ командный ПЛ', 'Франция командный ПЛ', 'Англия командный ПЛ', 'Испания командный',
             'Италия командный', 'Германия командный', 'ЛЧ командный', 'ЛЧ командный Лига2']:
        for team in tourn_teams:
            initial_ratings[team] += add_rating

print 'Initial top 20'
print_ratings(initial_ratings, n_best=20)

print 'Mean rating:', np.mean(initial_ratings.values())
best_k, _norm, e = validate_params(initial_ratings, np.linspace(10, 20, 30).tolist(), [best_norm])
print 'Best obtained error:', e
print 'K:', best_k, 'norm:', best_norm
# best_k = 10.

final_ratings, e, history = sim_season(match_results, initial_ratings, all_teams, best_k, best_norm)
print 'Final'
print_ratings(final_ratings)

ratings_series_, dates_, updates_, tournaments_ = history

plot_series(ratings_series_, dates_, [team for pl, team, rt in sorted_rating(final_ratings, n_best=11)])
plt.savefig(path + 'h2h-elo-images/_top11.png')

for pp, tt, rr in sorted_rating(final_ratings, n_best=10):
    plot_series(ratings_series_, dates_,
                [team for pl, team, rt in sorted_rating(final_ratings, n_best=10)], highlight=tt)
    plt.savefig(path + 'h2h-elo-images/{}.png'.format(tt))


for i, (pp, tt, rr) in enumerate(sorted_rating(final_ratings)[10:]):
    plot_series(ratings_series_, dates_,
                [team for pl, team, rt in sorted_rating(final_ratings, n_best=i + 15)[i + 5:]], highlight=tt)
    plt.savefig(path + 'h2h-elo-images/{}.png'.format(tt))
