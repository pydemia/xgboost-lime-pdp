# Env. Setting

## Prepare a Docker Image

## SET Docker virtual env

## Install `xgboost`: `xgboost4j-spark`

* set `JAVA_HOME`
```sh
export JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"
```

* clone `xgboost` from git (branch: `release_0.82`)
```sh
cd
mkdir git
cd git
git clone --recursive --single-branch --branch release_0.82 https://github.com/dmlc/xgboost
```

* Install `maven`
```sh
sudo apt-get update
sudo apt-get install maven
```

* Install `xgboost4j-spark`
```sh
cd ~/git/xgboost/jvm-packages/xgboost4j-spark
mvn -DskipTests install
```
