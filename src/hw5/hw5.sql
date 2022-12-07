-- Command for Mariadb to use baseball database for the queries below.
USE baseball;


-- Fix the column type issue on it, so the id is sorting as integer.
ALTER TABLE inning MODIFY COLUMN game_id INT UNSIGNED NOT NULL;

-- one game has impossible data for the temperature, not taking this game to consideration
DELETE FROM boxscore WHERE temp ='7882 degrees';

-- HomeTeamWins table: 1 as home team win, 0 as home team lose
DROP TABLE IF EXISTS game_info;
CREATE TABLE game_info
SELECT game_id,
       substr(temp,1, instr(temp, ' ') -1) AS temp,
       away_runs,
       away_errors,
       home_runs,
       home_errors,
       CASE WHEN winner_home_or_away="H" THEN 1 ELSE 0 END AS HomeTeamWins
FROM boxscore;

CREATE INDEX game_info_index ON game_info (game_id);


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
SELECT t.game_id,
       g.local_date,
       t.team_id AS home_team,
       t.plateApperance AS home_PA,
       (t.Hit / t.atBat) AS home_BA,
       t.Hit AS home_H,
       t.Home_Run AS home_HR,
       (t.Home_Run / 9) AS home_HR9,
       t.Walk AS home_BB,
       (t.Walk / 9) AS home_BB9,
       t.Strikeout AS home_K,
       (t.Strikeout / 9) AS home_K9,
       (t.Strikeout / NULLIF(t.Walk,0)) AS home_KBB,
       t.Triple_Play AS home_TP,
       t.Flyout AS home_Flyout,
       t.Grounded_Into_DP AS home_GIDP
FROM team_pitching_counts t
JOIN game_date g
ON t.game_id = g.game_id
WHERE homeTeam = 1;

CREATE UNIQUE INDEX ht_pitching_data_index
ON ht_pitching_data (game_id, home_team);
CREATE INDEX ht_id_index ON ht_pitching_data (game_id);


-- Select columns needed from team_pitching_counts [Away Team]
DROP TABLE IF EXISTS at_pitching_data;
CREATE TABLE at_pitching_data
SELECT t.game_id,
       g.local_date,
       t.team_id AS away_team,
       t.plateApperance AS away_PA,
       (t.Hit / t.atBat) AS away_BA,
       t.Hit AS away_H,
       t.Home_Run AS away_HR,
       (t.Home_Run / 9) AS away_HR9,
       t.Walk AS away_BB,
       (t.Walk / 9) AS away_BB9,
       t.Strikeout AS away_K,
       (t.Strikeout / 9) AS away_K9,
       (t.Strikeout / NULLIF(t.Walk,0)) AS away_KBB,
       t.Triple_Play AS away_TP,
       t.Flyout AS away_Flyout,
       t.Grounded_Into_DP AS away_GIDP
FROM team_pitching_counts t
JOIN game_date g
ON t.game_id = g.game_id
WHERE awayTeam = 1;

CREATE UNIQUE INDEX at_pitching_data_index
ON at_pitching_data (game_id, away_team);
CREATE INDEX at_id_index ON at_pitching_data (game_id);


-- joint home/away team table
DROP TABLE IF EXISTS ha_joint;
CREATE TABLE ha_joint
SELECT h.game_id,
       h.local_date,
       h.home_team,
       a.away_team,
       h.home_PA,
       h.home_BA,
       h.home_H,
       h.home_HR,
       h.home_HR9,
       h.home_BB,
       h.home_BB9,
       h.home_K,
       h.home_K9,
       h.home_KBB,
       h.home_TP,
       h.home_Flyout,
       h.home_GIDP,
       i.home_runs,
       i.home_errors,
       a.away_PA,
       a.away_BA,
       a.away_H,
       a.away_HR,
       a.away_HR9,
       a.away_BB,
       a.away_BB9,
       a.away_K,
       a.away_K9,
       a.away_KBB,
       a.away_TP,
       a.away_Flyout,
       a.away_GIDP,
       i.away_runs,
       i.away_errors,
       (h.home_PA - a.away_PA) AS diff_PA,
       (h.home_BA - a.away_BA) AS diff_BA,
       (h.home_H - a.away_H) AS diff_H,
       (h.home_HR - a.away_HR) AS diff_HR,
       (h.home_HR9 - a.away_HR9) AS diff_HR9,
       (h.home_BB - a.away_BB) AS diff_BB,
       (h.home_BB9 - a.away_BB9) AS diff_BB9,
       (h.home_K - a.away_K) AS diff_K,
       (h.home_K9 - a.away_K9) AS diff_K9,
       (h.home_KBB - a.away_KBB) AS diff_KBB,
       (h.home_TP - a.away_TP) AS diff_TP,
       (h.home_Flyout - a.away_Flyout) AS diff_Flyout,
       (h.home_GIDP - a.away_GIDP) AS diff_GIDP,
       (i.home_runs - i.away_runs) AS diff_runs,
       (i.home_errors - i.away_errors) AS diff_errors
FROM ht_pitching_data h
JOIN at_pitching_data a
ON h.game_id = a.game_id
JOIN game_info i
ON h.game_id = i.game_id
GROUP BY h.game_id
ORDER BY h.game_id, h.home_team, a.away_team ASC;


CREATE INDEX ha_id_index ON ha_joint (game_id);
CREATE INDEX ha_d_index ON ha_joint (local_date);


-- Rolling 100 for ha_joint
DROP TABLE IF EXISTS rolling_full;
CREATE TABLE rolling_full
SELECT h1.game_id,
       h1.local_date,
       h1.home_team,
       h1.away_team,
       COUNT(h2.game_id) AS num_games,
       SUM(h2.home_PA) / COUNT(h2.game_id) AS r_home_PA,
       SUM(h2.home_BA) / COUNT(h2.game_id) AS r_home_BA,
       SUM(h2.home_H) / COUNT(h2.game_id) AS r_home_H,
       SUM(h2.home_HR) / COUNT(h2.game_id) AS r_home_HR,
       SUM(h2.home_HR9) / COUNT(h2.game_id) AS r_home_HR9,
       SUM(h2.home_BB) / COUNT(h2.game_id) AS r_home_BB,
       SUM(h2.home_BB9) / COUNT(h2.game_id) AS r_home_BB9,
       SUM(h2.home_K) / COUNT(h2.game_id) AS r_home_K,
       SUM(h2.home_K9) / COUNT(h2.game_id) AS r_home_K9,
       SUM(h2.home_KBB) / COUNT(h2.game_id) AS r_home_KBB,
       SUM(h2.home_TP) / COUNT(h2.game_id) AS r_home_TP,
       SUM(h2.home_Flyout) / COUNT(h2.game_id) AS r_home_Flyout,
       SUM(h2.home_GIDP) / COUNT(h2.game_id) AS r_home_GIDP,
       SUM(h2.home_runs) / COUNT(h2.game_id) AS r_home_runs,
       SUM(h2.home_errors) / COUNT(h2.game_id) AS r_home_errors,
       SUM(h2.away_PA) / COUNT(h2.game_id) AS r_away_PA,
       SUM(h2.away_BA) / COUNT(h2.game_id) AS r_away_BA,
       SUM(h2.away_H) / COUNT(h2.game_id) AS r_away_H,
       SUM(h2.away_HR) / COUNT(h2.game_id) AS r_away_HR,
       SUM(h2.away_HR9) / COUNT(h2.game_id) AS r_away_HR9,
       SUM(h2.away_BB) / COUNT(h2.game_id) AS r_away_BB,
       SUM(h2.away_BB9) / COUNT(h2.game_id) AS r_away_BB9,
       SUM(h2.away_K) / COUNT(h2.game_id) AS r_away_K,
       SUM(h2.away_K9) / COUNT(h2.game_id) AS r_away_K9,
       SUM(h2.away_KBB) / COUNT(h2.game_id) AS r_away_KBB,
       SUM(h2.away_TP) / COUNT(h2.game_id) AS r_away_TP,
       SUM(h2.away_Flyout) / COUNT(h2.game_id) AS r_away_Flyout,
       SUM(h2.away_GIDP) / COUNT(h2.game_id) AS r_away_GIDP,
       SUM(h2.away_runs) / COUNT(h2.game_id) AS r_away_runs,
       SUM(h2.away_errors) / COUNT(h2.game_id) AS r_away_errors,
       SUM(h2.diff_PA) / COUNT(h2.game_id) AS r_diff_PA,
       SUM(h2.diff_BA) / COUNT(h2.game_id) AS r_diff_BA,
       SUM(h2.diff_H) / COUNT(h2.game_id) AS r_diff_H,
       SUM(h2.diff_HR) / COUNT(h2.game_id) AS r_diff_HR,
       SUM(h2.diff_HR9) / COUNT(h2.game_id) AS r_diff_HR9,
       SUM(h2.diff_BB) / COUNT(h2.game_id) AS r_diff_BB,
       SUM(h2.diff_BB9) / COUNT(h2.game_id) AS r_diff_BB9,
       SUM(h2.diff_K) / COUNT(h2.game_id) AS r_diff_K,
       SUM(h2.diff_K9) / COUNT(h2.game_id) AS r_diff_K9,
       SUM(h2.diff_KBB) / COUNT(h2.game_id) AS r_diff_KBB,
       SUM(h2.diff_TP) / COUNT(h2.game_id) AS r_diff_TP,
       SUM(h2.diff_Flyout) / COUNT(h2.game_id) AS r_diff_Flyout,
       SUM(h2.diff_GIDP) / COUNT(h2.game_id) AS r_diff_GIDP,
       SUM(h2.diff_runs) / COUNT(h2.game_id) AS r_diff_runs,
       SUM(h2.diff_errors) / COUNT(h2.game_id) AS r_diff_errors
FROM ha_joint h1
JOIN ha_joint h2
ON h1.home_team = h2.home_team
AND h2.local_date
    BETWEEN DATE_SUB(h1.local_date, INTERVAL 100 DAY)
    AND DATE_SUB(h1.local_date, INTERVAL 1 DAY)
GROUP BY h1.game_id
ORDER BY h1.game_id ASC;


CREATE INDEX r_f_id_index ON rolling_full (game_id);
CREATE INDEX r_f_date_index ON rolling_full (local_date);


-- Table for model fitting
DROP TABLE IF EXISTS model_data;
CREATE TABLE model_data
SELECT A.game_id,
       A.local_date,
       A.home_team,
       A.away_team,
       A.r_home_PA,
       A.r_home_BA,
       A.r_home_H,
       A.r_home_HR,
       A.r_home_HR9,
       A.r_home_BB,
       A.r_home_BB9,
       A.r_home_K,
       A.r_home_K9,
       A.r_home_KBB,
       A.r_home_TP,
       A.r_home_Flyout,
       A.r_home_GIDP,
       A.r_home_runs,
       A.r_home_errors,
       A.r_away_PA,
       A.r_away_BA,
       A.r_away_H,
       A.r_away_HR,
       A.r_away_HR9,
       A.r_away_BB,
       A.r_away_BB9,
       A.r_away_K,
       A.r_away_K9,
       A.r_away_KBB,
       A.r_away_TP,
       A.r_away_Flyout,
       A.r_away_GIDP,
       A.r_away_runs,
       A.r_away_errors,
       A.r_diff_PA,
       A.r_diff_BA,
       A.r_diff_H,
       A.r_diff_HR,
       A.r_diff_HR9,
       A.r_diff_BB,
       A.r_diff_BB9,
       A.r_diff_K,
       A.r_diff_K9,
       A.r_diff_KBB,
       A.r_diff_TP,
       A.r_diff_Flyout,
       A.r_diff_GIDP,
       A.r_diff_runs,
       A.r_diff_errors,
       R.temp,
       R.HomeTeamWins
FROM rolling_full A
JOIN game_info R
ON A.game_id = R.game_id
GROUP BY game_id
ORDER BY A.game_id, A.local_date;
