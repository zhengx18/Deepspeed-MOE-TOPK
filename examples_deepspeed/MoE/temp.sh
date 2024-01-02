# echo $MASTER_ADDR
# ping 10.82.120.153
export http_proxy=http://oversea-squid2.ko.txyun:11080 https_proxy=http://oversea-squid2.ko.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
apt-get install bc

# DATA_PERCENT=0.5
# TRAIN_TOKENS=$(echo "100000000000 * $DATA_PERCENT" | bc)
# TRAIN_TOKENS=$(printf "%.0f" $TRAIN_TOKENS)
# echo TRAIN_TOKENS ${TRAIN_TOKENS}
# # TRAIN_TOKENS=330000000000
# MASTER_ADDR=10.82.124.33
# MASTER_PORT=2333
# echo $MASTER_ADDR
# echo $MASTER_PORT
# ping 10.82.124.33