-- Command for Mariadb to use baseball database for the queries below.
USE baseball;

-- Fix the column type issue on it, so the id is sorting as integer.
ALTER TABLE inning MODIFY COLUMN game_id INT UNSIGNED NOT NULL;


-- Calculate the batting average using SQL queries for every player
-- batting average = Hit / atBat
-- Annual
DROP TABLE IF EXISTS annual_bat_avg;

CREATE TABLE annual_bat_avg
SELECT B.batter, YEAR(G.local_date) AS Game_Year, ROUND(SUM(B.Hit)/(SUM(B.atBat)),3) AS Annual_Batting_Avg
FROM batter_counts B
JOIN game G
ON B.game_id = G.game_id
WHERE B.atBat > 0
GROUP BY B.batter, Game_Year
ORDER BY B.batter, Game_Year;

-- Historical
DROP TABLE IF EXISTS hist_bat_avg;

CREATE TABLE hist_bat_avg
SELECT batter, ROUND(SUM(Hit)/(SUM(atBat)),3) AS Historical_Batting_Avg
FROM batter_counts
WHERE atBat > 0
GROUP BY batter
ORDER BY batter;


-- Intermediary table batter_game
DROP TABLE IF EXISTS batter_game;

CREATE TABLE batter_game
SELECT b.batter
       , g.game_id
       , DATE(g.local_date) AS local_date
       , b.Hit, b.atBat
FROM batter_counts b
JOIN game g
ON b.game_id = g.game_id
ORDER BY b.batter;

CREATE INDEX bg_index_batter
ON batter_game (batter);

CREATE INDEX bg_index_game_id
ON batter_game (game_id);

-- Look at the last 100 days that player was in prior to this game
-- Rolling: 100 days prior without the day of the game
-- Using 'DATE_SUB' to get the date of 100 day prior to the game local date
DROP TABLE IF EXISTS diff_100_stats;

CREATE TABLE diff_100_stats
SELECT b1.batter
       , DATE(b1.local_date) AS local_date
       , SUM(b2.Hit)/NULLIF(SUM(b2.atBat),0) AS roll_100_stat
FROM batter_game b1
JOIN batter_game b2
ON b1.batter = b2.batter
AND b2.local_date BETWEEN DATE_SUB(b1.local_date, INTERVAL 100 DAY) AND DATE_SUB(b1.local_date, INTERVAL 1 DAY)
WHERE b2.atBat > 0
GROUP BY b1.batter, b1.local_date;

CREATE INDEX rolling_index ON diff_100_stats(batter, local_date);

-- Final rolling table
DROP TABLE IF EXISTS avg_100_stats;

CREATE TABLE avg_100_stats
SELECT b.batter, b.game_id, d.roll_100_stat
FROM batter_game b
JOIN diff_100_stats d
ON b.batter = d.batter
AND DATE(b.local_date) = d.local_date
GROUP BY b.batter, b.game_id;
