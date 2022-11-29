-- Command for Mariadb to use baseball database for the queries below.
USE baseball;


-- Fix the column type issue on it, so the id is sorting as integer.
ALTER TABLE inning MODIFY COLUMN game_id INT UNSIGNED NOT NULL;


-- HomeTeamWins table: 1 as home team win, 0 as home team lose
DROP TABLE IF EXISTS game_result;
CREATE TABLE game_result
SELECT game_id,
       CASE WHEN winner_home_or_away="H" THEN 1 ELSE 0 END AS HomeTeamWins
FROM boxscore;

CREATE INDEX game_result_index ON game_result (game_id);


-- Select columns needed from game table
DROP TABLE IF EXISTS game_date;
CREATE TABLE game_date
SELECT game_id,
       DATE(local_date) AS local_date
FROM game;

CREATE INDEX game_date_index ON game_date (game_id);

-- Select columns needed from team_pitching_counts [Home Team]
DROP TABLE IF EXISTS ht_pitching_data;
CREATE TABLE ht_pitching_data
SELECT game_id,
       team_id AS home_team,
       plateApperance AS home_PA,
       (Hit / atBat) AS home_BA,
       Hit AS home_H,
       Home_Run AS home_HR,
       Walk AS home_BB,
       Strikeout AS home_K,
       Triple_Play AS home_TP,
       Flyout AS home_Flyout,
       Grounded_Into_DP AS home_GIDP
FROM team_pitching_counts
WHERE homeTeam = 1;

CREATE UNIQUE INDEX ht_pitching_data_index
ON ht_pitching_data (game_id, home_team);
CREATE INDEX ht_id_index ON ht_pitching_data (game_id);


-- Select columns needed from team_pitching_counts [Away Team]
DROP TABLE IF EXISTS at_pitching_data;
CREATE TABLE at_pitching_data
SELECT game_id,
       team_id AS away_team,
       plateApperance AS away_PA,
       (Hit / atBat) AS away_BA,
       Hit AS away_H,
       Home_Run AS away_HR,
       Walk AS away_BB,
       Strikeout AS away_K,
       Triple_Play AS away_TP,
       Flyout AS away_Flyout,
       Grounded_Into_DP AS away_GIDP
FROM team_pitching_counts
WHERE awayTeam = 1;

CREATE UNIQUE INDEX at_pitching_data_index
ON at_pitching_data (game_id, away_team);
CREATE INDEX at_id_index ON at_pitching_data (game_id);

-- Final table for model fitting
DROP TABLE IF EXISTS model_data;
CREATE TABLE model_data
SELECT H.game_id,
       H.home_team,
       H.home_PA,
       H.home_BA,
       H.home_H,
       H.home_HR,
       H.home_BB,
       H.home_K,
       H.home_TP,
       H.home_Flyout,
       H.home_GIDP,
       A.away_team,
       A.away_PA,
       A.away_BA,
       A.away_H,
       A.away_HR,
       A.away_BB,
       A.away_K,
       A.away_TP,
       A.away_Flyout,
       A.away_GIDP,
       G.local_date,
       R.HomeTeamWins
FROM ht_pitching_data H
JOIN at_pitching_data A
ON H.game_id = A.game_id
JOIN game_date G
ON H.game_id = G.game_id
JOIN game_result R
ON H.game_id = R.game_id
GROUP BY game_id
ORDER BY H.game_id, G.local_date;

