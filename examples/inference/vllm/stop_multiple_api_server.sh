ps -ef | grep api_server | grep -v grep | awk '{ print $2 }' | xargs kill
