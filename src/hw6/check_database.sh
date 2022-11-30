#!/bin/bash
sleep 100
if mariadb -uroot -proot --host="mariadb" -e "USE baseball;"

then
    echo "Baseball exists in mariadb."
    mariadb -uroot -proot --host="mariadb" baseball < /src/hw6/hw6.sql
    # write result in a file
    mariadb -uroot -poot --host="mariadb" baseball -e "SELECT * FROM rolling_12560;" > /src/hw6/rba_results.txt
else
    echo "Baseball does not exist in mariadb."
    mariadb -uroot -proot --host="mariadb" -e "CREATE DATABASE baseball;"
    mariadb -uroot -proot --host="mariadb" baseball < /src/baseball.sql
fi
