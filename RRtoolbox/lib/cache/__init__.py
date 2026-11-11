# -*- coding: utf-8 -*-
"""
.. moduleauthor:: David Toro <davsamirtor@gmail.com>

:platform: Unix, Windows
:synopsis: Serialize and Memoize.

Contains memoizing, caching, serializing and memory-mapping methods so as to let the package
save its state (persistence) and to let a method "remember" what it processed in a session (with cache) or
between sessions (memoization and serialization) of the same input content once processed. It also wraps mmapping
functions to let objects "live" on disk (slower but almost unlimited) rather than in memory (faster but limited).

*@cache* is used as replacement of *@property* to compute a class method once.
It is computed only one time after which an attribute of the same name is generated in its place.

*@cachedProperty* is used as replacement of *@property* to compute
a class method depending on changes in its watched variables.

*@memoize* used as a general memoizer decorator for functions
where metadata is generated to disk for persistence.

Made by Davtoh, powered by joblib.
Dependent project: https://github.com/joblib/joblib
"""

# Re-export root persistence and contract exceptions
try:
    from ..root import (NotCallable, NotCreatable, VariableNotSettable,
                       VariableNotDeletable, CorruptPersistent, secure_open)
except (ImportError, ValueError):
    from RRtoolbox.lib.root import (NotCallable, NotCreatable, VariableNotSettable,
                                   VariableNotDeletable, CorruptPersistent, secure_open)

# Submodule re-exports
from .lazy import ObjectGetter, Retriever, LazyDict
from .memoize import (
    NotMemorizedFunc,
    MemorizedFunc,
    DynamicMemoizedFunc,
    Memory,
    Memoizer,
    memoize,
    cache,
    cachedProperty,
    Memo,
)
from .resource_manager import ResourceManager
from .disk_dict import mapper, MemoizedDict

__all__ = [
    # Exceptions & primitives from root
    "NotCallable",
    "NotCreatable",
    "VariableNotSettable",
    "VariableNotDeletable",
    "CorruptPersistent",
    "secure_open",
    # Lazy & object getters
    "ObjectGetter",
    "Retriever",
    "LazyDict",
    # Memoizers & caching decorators
    "NotMemorizedFunc",
    "MemorizedFunc",
    "DynamicMemoizedFunc",
    "Memory",
    "Memoizer",
    "memoize",
    "cache",
    "cachedProperty",
    "Memo",
    # Resource management
    "ResourceManager",
    # Persistent disk storage & mapping
    "mapper",
    "MemoizedDict",
]

