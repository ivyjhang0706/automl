#!/bin/bash
set -e

# root 預設沒有密碼、無法用密碼登入 SSH。第一次啟動 container 後，
# 自己手動跑一次：docker compose exec automl passwd root
/usr/sbin/sshd

exec "$@"
