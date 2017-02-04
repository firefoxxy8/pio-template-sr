# Survival Regression Template

This PredictionIO template is based on AFT (accelerated failure time) algorithm described in [MLlib - Survival regression](https://spark.apache.org/docs/1.6.1/ml-classification-regression.html#survival-regression) and in [API](https://spark.apache.org/docs/1.6.1/api/java/index.html?org/apache/spark/ml/regression/AFTSurvivalRegression.html)

## Prerequisites
Template requires Spark 1.6.1 and PredictionIO 0.10.0-incubating.
Prebuilt docker image with required versions is available [here](https://hub.docker.com/r/goliasz/docker-predictionio-dev/) 

## Deployment
```
pio template get goliasz/pio-template-sr --version "0.4" sr1
cd sr1
pio build --verbose
pio app new sr1 --access-key 1234
sh data/import_test.sh <<APP_ID>>
nano engine.json <-- set APP_NAME to sr1
pio train
pio deploy --port 8000 &
```
## Test
```
curl -i -X POST http://localhost:8000/queries.json -H "Content-Type: application/json" -d '{"features":[1.560,-0.605]}'
```
Should give in result
```
{
  "coefficients": [
    -0.2633608588194104, 
    0.22152319227842276
  ], 
  "intercept": 2.6380946151040012, 
  "prediction": 5.718979487634966, 
  "quantiles": [
    1.1603238947151593, 
    4.995456010274735
  ], 
  "scale": 1.5472345574364683
}
```

## Compatibility 

0.5.0 - PIO v0.10.0-incubating

0.4.1 - PIO v0.9.5

## License
This Software is licensed under the Apache Software Foundation version 2 licence found here: http://www.apache.org/licenses/LICENSE-2.0

