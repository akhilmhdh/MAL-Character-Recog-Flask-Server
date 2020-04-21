# MAL-Character-Recog-Flask-Server

This project is used for hosting the DL pytorch model of malayalam character recognition. Made in Flask a python framework for backend. Really simple and neat.

## Getting Started

Unix Bash (Linux, Mac, etc.):

```
$ export FLASK_APP=app
$ export FLASK_ENV=development
$ flask run
```

Windows CMD:

```
> set FLASK_APP=app
> set FLASK_ENV=development
> flask run
```

Windows PowerShell:

```
> $env:FLASK_APP = "app"
> $env:FLASK_ENV = "development"
> flask run
```

### Prerequisites

```
FLASK
TORCH
TORCHVISION
NUMPY
```

## Deployment

Currently deployed to heroku using gunicorn. Requirement text is provided with the versions of libraries used. 

## Built With

* [FLASK](https://flask.palletsprojects.com/en/1.1.x/) - The web framework used
* [TORCH](https://pytorch.org/) - The DL model behind it
* [HEROKU CLI](https://devcenter.heroku.com/articles/heroku-cli) - Used for CI in Heroku 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Pytorch documentation 
* Heroku simplicity loved it 

