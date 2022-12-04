-- Command for Mariadb to use baseball database for the queries below.
USE baseball;

-- Fix the column type issue on it, so the id is sorting as integer.
ALTER TABLE inning MODIFY COLUMN game_id INT UNSIGNED NOT NULL;

-- Create intermediary table
DROP TABLE IF EXISTS batter_game;

CREATE TABLE batter_game
SELECT bc.batter
       , bc.game_id
       , DATE(g.local_date) AS local_date
       ,bc.atBat
       ,bc.Hit
FROM batter_counts bc
JOIN game g
ON bc.game_id = g.game_id;

CREATE UNIQUE INDEX batter_game_index ON batter_game (batter, game_id);
CREATE INDEX bg1_index ON batter_game (batter);
CREATE INDEX bg2_index ON batter_game (game_id);


-- Create table for game id 12560 rolling 100.
DROP TABLE IF EXISTS rolling_12560;

CREATE TABLE rolling_12560
SELECT bg1.batter
       , bg1.game_id
       , bg1.local_date
       , SUM(bg2.Hit)/NULLIF(SUM(bg2.atBat),0) AS rolling_ba
FROM batter_game bg1
JOIN batter_game bg2
ON bg1.batter = bg2.batter
AND bg2.local_date
    BETWEEN DATE_SUB(bg1.local_date, INTERVAL 100 DAY)
    AND DATE_SUB(bg1.local_date, INTERVAL 1 DAY)
WHERE bg1.game_id = 12560
GROUP BY bg1.batter
ORDER BY bg1.batter ASC;