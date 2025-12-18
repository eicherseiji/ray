from ray._private.runtime_env.image_uri import ImageURIPlugin


def get_image_uri_plugin_cls():
    return ImageURIPlugin


# Anyscale overrides


def get_image_uri_plugin_cls():  # noqa: F811
    from ray.anyscale._private.runtime_env.image_uri import AnyscaleImageURIPlugin

    return AnyscaleImageURIPlugin
