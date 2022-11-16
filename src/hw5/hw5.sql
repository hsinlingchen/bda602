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


-- Pitching data with Starting Pitcher's Data
DROP TABLE IF EXISTS pitching_data;
CREATE TABLE pitching_data
SELECT P.game_id,
       P.team_id,
       P.homeTeam,
       P.awayTeam,
       P.startingPitcher,
       (P.endingInning - P.startingInning + 1) AS IP,
       P.Hit AS H,
       P.Grounded_Into_DP AS GIDP,
       P.Home_Run AS HR,
       P.Strikeout AS K,
       P.Walk AS BB,
       P.pitchesThrown AS PIT,
       P.Triple_Play AS TP,
       ((P.Walk + P.Hit) / (P.endingInning - P.startingInning + 1)) AS WHIP,
       ((P.Strikeout + P.Walk) / (P.endingInning - P.startingInning + 1)) AS PFR,
       G.HomeTeamWins
FROM pitcher_counts P
JOIN game_result G
ON P.game_id = G.game_id
WHERE P.startingPitcher = 1;


