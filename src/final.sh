#!/bin/bash
sleep 99

if ! mariadb -h mariadb -u root -ppassword123 -e "USE baseball;"
then
  echo "Baseball does not exist in mariadb."
  mariadb -h mariadb -u root -ppassword123 -e "CREATE DATABASE baseball;"
  mariadb -h mariadb -u root -ppassword123 baseball < baseball.sql
fi

echo "Baseball exists in mariadb."
mariadb -h mariadb -u root -ppassword123 baseball < hw5.sql
python hw5.py
echo "Result has been populated."
