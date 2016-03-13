# pio-template-sr
Survival Regression Template

This PredictionIO template is based on AFT (accelerated failure time) algorithm described in [MLlib - Survival regression](https://spark.apache.org/docs/1.6.1/ml-classification-regression.html#survival-regression)

## Prerequisites
Template requires Spark 1.6.1 and PredictionIO 0.9.5.
Prebuilt docker image with required versions is available [here](https://hub.docker.com/r/goliasz/docker-predictionio-dev/) 

## Deployment
```
pio template get goliasz/pio-template-sr --version "0.1" sr1
cd sr1
pio build --verbose
pio app new sr1
sh data/import_test.sh <<APP_ID>>
nano engine.json <-- set APP_NAME to sr1
pio train
pio deploy --port 8000 &
```
## Test
```
curl -i -X POST http://localhost:8000/queries.json -H "Content-Type: application/json" -d '{"features":[0.1, 0.2, 0.3]}'
```
Should give in result
```
{"TODO"}
```

## License
This Software is licensed under the Apache Software Foundation version 2 licence found here: http://www.apache.org/licenses/LICENSE-2.0

