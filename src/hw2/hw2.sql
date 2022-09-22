-- Command for Mariadb to use baseball database for the queries below.
USE baseball;

-- Fix the column type issue on it, so the id is sorting as integer.
ALTER TABLE inning MODIFY COLUMN game_id INT UNSIGNED NOT NULL;


-- Calculate the batting average using SQL queries for every player
-- Annual
DROP TABLE IF EXISTS annual_bat_avg;
-- NULLIF(SUM(atBAt),0) to avoid division by zero error
CREATE TABLE annual_bat_avg
SELECT B.batter, YEAR(G.local_date) AS Game_Year, ROUND(SUM(B.Hit)/NULLIF(SUM(B.atBat),0),3) AS Annual_Batting_Avg
FROM batter_counts B
JOIN game G
ON B.game_id = G.game_id
GROUP BY B.batter, Game_Year
ORDER BY B.batter, Game_Year;

-- Historical
DROP TABLE IF EXISTS Historical_Batting_Avg;

CREATE TABLE hist_bat_avg
SELECT batter, ROUND(SUM(Hit)/NULLIF(SUM(atBat),0),3) AS Historical_Batting_Avg
FROM batter_counts
GROUP BY batter
ORDER BY batter;

-- Rolling (over last 100 days)
DROP TABLE IF EXISTS rolling_100;

-- Look at the last 100 days that player was in prior to this game
-- Rolling: 100 days prior without the day of the game
-- USING 'DATE_SUB' to get the date of 100 day prior to the game local date
-- BETWEEN includes both begin and end days
CREATE TABLE rolling_100
SELECT B.batter, G.game_id, G.local_date, DATE_SUB(G.local_date, INTERVAL 100 DAY) AS Date_100days_prior
       ,(SELECT ROUND(SUM(BC.Hit)/NULLIF(SUM(BC.atBat),0),3)
         FROM batter_counts BC
         JOIN game Ga
         ON BC.game_id = Ga.game_id
         WHERE (Ga.local_date BETWEEN DATE_SUB(G.local_date, INTERVAL 100 DAY) AND DATE_SUB(G.local_date, INTERVAL 1 DAY))
         AND (BC.batter=B.batter)
         GROUP BY BC.batter) AS Rolling_100_Avg
FROM batter_counts B
JOIN game G
ON B.game_id = G.game_id
GROUP BY B.batter, G.game_id
ORDER BY B.batter, G.game_id;




