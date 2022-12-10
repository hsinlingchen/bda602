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
       CASE WHEN winner_home_or_away="H" THEN 1 ELSE 0 END AS HomeTeamWins,
       CASE WHEN winner_home_or_away="A" THEN 1 ELSE 0 END AS HomeTeamLoses
FROM boxscore;

CREATE INDEX game_info_index ON game_info (game_id);

-- Select columns needed from game table
DROP TABLE IF EXISTS game_date;
CREATE TABLE game_date
SELECT game_id,
       DATE(local_date) AS local_date
FROM game;

CREATE INDEX game_date_index ON game_date (game_id);

-- Create Table for all pitching
DROP TABLE IF EXISTS all_pitching;
CREATE TABLE all_pitching
SELECT t.game_id,
       d.local_date,
       t.team_id,
       t.homeTeam,
       t.win,
       t.plateApperance AS PA,
       t.atBat,
       t.Hit AS H,
       t.Hit / NULLIF(t.atBat,0) AS BA,
       t.Home_Run AS HR,
       (t.Home_Run / 9) AS HR9,
       t.Walk AS BB,
       (t.Walk / 9) AS BB9,
       t.Strikeout AS K,
       (t.Strikeout / 9) AS K9,
       (t.Strikeout / NULLIF(t.Walk,0)) AS KBB,
       t.Double_Play AS DP,
       t.Triple_Play AS TP,
       t.Flyout AS Flyout,
       t.Grounded_Into_DP AS GIDP,
       t.Fan_interference AS FI,
       t.Field_Error AS FE,
       i.HomeTeamWins,
       i.HomeTeamLoses
FROM team_pitching_counts t
JOIN game_date d
ON t.game_id = d.game_id
JOIN game_info i
ON t.game_id = i.game_id
GROUP BY t.game_id, t.team_id;

CREATE UNIQUE INDEX all_pitching_index
ON all_pitching (game_id, team_id);
CREATE INDEX all_pitching_id_index ON all_pitching (game_id);
CREATE INDEX all_pitching_tid_index ON all_pitching (team_id);

-- Rolling Table for all pitching data
DROP TABLE IF EXISTS r_all_pitching;
CREATE TABLE r_all_pitching
SELECT a1.game_id,
       a1.local_date,
       a1.team_id,
       a1.homeTeam,
       a1.HomeTeamWins,
       COUNT(a2.game_id) AS num_games,
       SUM(a2.PA) / COUNT(a2.game_id) AS r_PA,
       SUM(a2.H) / COUNT(a2.game_id) AS r_H,
       SUM(a2.BA) / COUNT(a2.game_id) AS r_BA,
       SUM(a2.HR) / COUNT(a2.game_id) AS r_HR,
       SUM(a2.HR9) / COUNT(a2.game_id) AS r_HR9,
       SUM(a2.BB) / COUNT(a2.game_id) AS r_BB,
       SUM(a2.BB9) / COUNT(a2.game_id) AS r_BB9,
       SUM(a2.K) / COUNT(a2.game_id) AS r_K,
       SUM(a2.K9) / COUNT(a2.game_id) AS r_K9,
       SUM(a2.KBB) / COUNT(a2.game_id) AS r_KBB,
       SUM(a2.Flyout) / COUNT(a2.game_id) AS r_Flyout,
       SUM(a2.DP) / COUNT(a2.game_id) AS r_DP,
       SUM(a2.TP) / COUNT(a2.game_id) AS r_TP,
       SUM(a2.GIDP) / COUNT(a2.game_id) AS r_GIDP,
       SUM(a2.FI) / COUNT(a2.game_id) AS r_FI,
       SUM(a2.FE) / COUNT(a2.game_id) AS r_FE,
       SUM(a2.win) / COUNT(a2.game_id) AS r_WP
FROM all_pitching a1
JOIN all_pitching a2
ON a1.team_id = a2.team_id
AND a2.local_date BETWEEN DATE_SUB(a1.local_date, INTERVAL 60 DAY) AND DATE_SUB(a1.local_date, INTERVAL 1 DAY)
GROUP BY a1.team_id, a1.game_id
ORDER BY a1.game_id;

CREATE UNIQUE INDEX r_all_pitching_index
ON r_all_pitching (game_id, team_id);
CREATE INDEX r_all_pitching_id_index ON r_all_pitching (game_id);
CREATE INDEX r_all_pitching_tid_index ON r_all_pitching (team_id);

-- Create table for home team
DROP TABLE IF EXISTS r_all_home;
CREATE TABLE r_all_home
SELECT * FROM r_all_pitching
WHERE homeTeam = 1;

CREATE UNIQUE INDEX r_all_home_index
ON r_all_home (game_id, team_id);
CREATE INDEX r_all_home_id_index ON r_all_home (game_id);
CREATE INDEX r_all_home_tid_index ON r_all_home (team_id);

-- Create table for away team
DROP TABLE IF EXISTS r_all_away;
CREATE TABLE r_all_away
SELECT * FROM r_all_pitching
WHERE homeTeam = 0;

CREATE UNIQUE INDEX r_all_away_index
ON r_all_away (game_id, team_id);
CREATE INDEX r_all_away_id_index ON r_all_away (game_id);
CREATE INDEX r_all_away_tid_index ON r_all_away (team_id);

-- Create Table for Model data
DROP TABLE IF EXISTS model_data;
CREATE TABLE model_data
SELECT h.game_id,
       h.local_date,
       h.team_id AS home_team,
       h.HomeTeamWins,
       h.r_PA AS r_home_PA,
       h.r_H AS r_home_H,
       h.r_BA AS r_home_BA,
       h.r_HR AS r_home_HR,
       h.r_HR9 AS r_home_HR9,
       h.r_BB AS r_home_BB,
       h.r_BB9 AS r_home_BB9,
       h.r_K AS r_home_K,
       h.r_K9 AS r_home_K9,
       h.r_KBB AS r_home_KBB,
       h.r_Flyout AS r_home_Flyout,
       h.r_DP AS r_home_DP,
       h.r_TP AS r_home_TP,
       h.r_GIDP AS r_home_GIDP,
       h.r_FI AS r_home_FI,
       h.r_FE AS r_home_FE,
       h.r_WP AS r_home_WP,
       a.team_id AS away_team,
       a.r_PA AS r_away_PA,
       a.r_H AS r_away_H,
       a.r_BA AS r_away_BA,
       a.r_HR AS r_away_HR,
       a.r_HR9 AS r_away_HR9,
       a.r_BB AS r_away_BB,
       a.r_BB9 AS r_away_BB9,
       a.r_K AS r_away_K,
       a.r_K9 AS r_away_K9,
       a.r_KBB AS r_away_KBB,
       a.r_Flyout AS r_away_Flyout,
       a.r_DP AS r_away_DP,
       a.r_TP AS r_away_TP,
       a.r_GIDP AS r_away_GIDP,
       a.r_FI AS r_away_FI,
       a.r_FE AS r_away_FE,
       a.r_WP AS r_away_WP,
       (h.r_PA - a.r_PA) AS r_diff_PA,
       (h.r_H - a.r_H) AS r_diff_H,
       (h.r_BA - a.r_BA) AS r_diff_BA,
       (h.r_HR - a.r_HR) AS r_diff_HR,
       (h.r_HR9 - a.r_HR9) AS r_diff_HR9,
       (h.r_BB - a.r_BB) AS r_diff_BB,
       (h.r_BB9 - a.r_BB9) AS r_diff_BB9,
       (h.r_K - a.r_K) AS r_diff_K,
       (h.r_K9 - a.r_K9) AS r_diff_K9,
       (h.r_KBB - a.r_KBB) AS r_diff_KBB,
       (h.r_Flyout - a.r_Flyout) AS r_diff_Flyout,
       (h.r_DP - a.r_DP) AS r_diff_DP,
       (h.r_TP - a.r_TP) AS r_diff_TP,
       (h.r_GIDP - a.r_GIDP) AS r_diff_GIDP,
       (h.r_FI - a.r_FI) AS r_diff_FI,
       (h.r_FE - a.r_FE) AS r_diff_FE,
       (h.r_WP - a.r_WP) AS r_diff_WP,
       i.temp
FROM r_all_home h
JOIN r_all_away a
ON h.game_id = a.game_id
JOIN game_info i
ON h.game_id = i.game_id
GROUP BY h.game_id
ORDER BY h.game_id;


CREATE UNIQUE INDEX md_index
ON model_data (game_id, home_team, away_team);
CREATE INDEX md_id_index ON model_data (game_id);
CREATE INDEX md_hid_index ON model_data (home_team);
CREATE INDEX md_aid_index ON model_data (away_team);

