class Meta(type):
    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        if not hasattr(cls, "registory"):
            # this is the base class
            cls.registory = {}
        else:
            # this is the subclass
            cls.registory[name] = cls


class Model(metaclass=Meta):
    pass
