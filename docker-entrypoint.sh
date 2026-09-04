#!/bin/bash
set -e

# root 登入已關閉，只能用各自帳號的金鑰登入（見 Dockerfile 的 pubkeys/ 設定）
/usr/sbin/sshd

# /share 是 runtime 才 mount 進來的，image build 時看不到、管不到權限，每次開機補設一次，
# 讓 automl 群組的人都能讀寫，且新建的檔案/資料夾自動繼承 automl 群組（setgid）
if [ -d /share/automl ]; then
    chgrp automl /share/automl 2>/dev/null || true
    chmod g+rwxs /share/automl 2>/dev/null || true
fi

exec "$@"
