#!/bin/bash
sleep 200

if ! mariadb -h mariadb -u root -ppassword123 -e "USE baseball;"
then
  echo "Baseball does not exist in mariadb."
  mariadb -h mariadb -u root -ppassword123 -e "CREATE DATABASE baseball;"
  mariadb -h mariadb -u root -ppassword123 baseball < ./baseball.sql
fi

echo "Baseball exists in mariadb."
mariadb -h mariadb -u root -ppassword123 baseball < /src/hw6/hw6.sql
# write result in a file
mariadb -h mariadb -u root -ppassword123 baseball -e "SELECT * FROM rolling_12560;" > /src/hw6/rba_results.txt
