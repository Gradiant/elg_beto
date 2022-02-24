# elg_beto
elg_beto is a tool that predicts which word fits in a masked space on a text using the maskes language model BETO.
This repository contains a dockerized API built over BETO for integrate it into the ELG. Its original code can
be found [here](https://github.com/dccuchile/beto).

## Install

```
sh docker-build.sh
```

## Execute
```
docker run --rm -p 0.0.0.0:8866:8866 --name beto elg_beto:1.0
```
## Use

```
curl -X POST  http://0.0.0.0:8866/predict_json -H 'Content-Type: application/json' -d '{"type": "text", "content":"El tribunal considera provado que los acusados han [MASK] por lo menos 24 millones de euros."}'
```


# Test
In the folder `test` you have the files for testing the API according to the ELG specifications.
It uses an API that acts as a proxy with your dockerized API that checks both the requests and the responses.
For this follow the instructions:
1) Configure the .env file with the data of the image and your API
2) Launch the test: `docker-compose up`
3) Make the requests, instead of to your API's endpoint, to the test's endpoint:
   ```
   curl -X POST  http://0.0.0.0:8866/processText/service -H 'Content-Type: application/json' -d '{"type": "text", "content":"El tribunal considera provado que los acusados han [MASK] por lo menos 24 millones de euros."}'
   ```
4) If your request and the API's response is compliance with the ELG API, you will receive the response.
   1) If the request is incorrect: Probably you will don't have a response and the test tool will not show any message in logs.
   2) If the response is incorrect: You will see in the logs that the request is proxied to your API, that it answers, but the test tool does not accept that response. You must analyze the logs.


## Citations:
The original work of this tool is:
- Canete, J., Chaperon, G., Fuentes, R., Ho, J. H., Kang, H., & Pérez, J. (2020). Spanish pre-trained bert model and evaluation data. Pml4dc at iclr, 2020, 2020.
- https://github.com/dccuchile/beto