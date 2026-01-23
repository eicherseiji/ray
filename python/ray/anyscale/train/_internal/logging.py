import logging.config

from ray.train.v2._internal.execution.context import TrainContext, TrainRunContext
from ray.train.v2._internal.logging.logging import LoggingManager


class AnyscaleLoggingManager(LoggingManager):
    @staticmethod
    def configure_controller_logger(context: TrainRunContext) -> None:
        """
        Configure the logger on the controller process, which is the `ray.train`
        and the `ray.anyscale` logger.
        """
        config = LoggingManager._get_controller_logger_config_dict(context)
        config["loggers"]["ray.anyscale.train"] = {
            "level": "INFO",
            "handlers": [
                "file_train_sys_controller",
                "file_train_app_controller",
                "console",
            ],
            "propagate": False,
        }
        logging.config.dictConfig(config)

    @staticmethod
    def configure_worker_logger(context: TrainContext) -> None:
        """
        Configure the loggers on the worker process, which contains the
        `ray.train` logger, the `ray.anyscale` logger, and the root logger.
        """
        config = LoggingManager._get_worker_logger_config_dict(context)
        config["loggers"]["ray.anyscale.train"] = {
            "level": "INFO",
            "handlers": ["file_train_sys_worker", "file_train_app_worker", "console"],
            "propagate": False,
        }
        logging.config.dictConfig(config)
