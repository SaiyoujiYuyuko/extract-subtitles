from typing import Callable, TypeVar, Generic, Any
from typing import Iterable, Iterator, List, Tuple, Dict

from re import findall
from itertools import chain, repeat, islice
from sys import stderr

T = TypeVar("T")
R = TypeVar("R", contravariant=True)
K = TypeVar("K")
V = TypeVar("V")

# == Normal function utils ==
identity = lambda x: x
noOp = lambda x: ()

def let(transform: Callable[[T], R], x: T) -> R:
  return transform(x) if x != None else None

def also(op: Callable[[T], Any], x: T) -> T:
  op(x)
  return x

def require(p: bool, message: str):
  if not p: raise ValueError(message)

def toMapper(transform: Callable[[T], R]) -> Callable[[Iterable[T]], List[R]]:
  return lambda xs: [transform(x) for x in xs]

def zipWithNext(xs: List[T]) -> Iterator[Tuple[T, T]]:
  require(len(xs) >= 2, f"len {len(xs)} is too short (< 2)")
  for i in range(1, len(xs)):
    yield (xs[i-1], xs[i])

def chunked(n: int, xs: Iterator[T]) -> Iterator[T]:
  while True:
    try: #< must return when inner gen finished
      first = next(xs)
    except StopIteration: return
    chunk = islice(xs, n)
    yield chain((first,) , chunk)

def collect2(selector2: Callable[[T], Tuple[R, R]], xs: Iterable[T]) -> Tuple[List[R], List[R]]:
  bs, cs = [], []
  for x in xs:
    b, c = selector2(x)
    bs.append(b); cs.append(c)
  return (bs, cs)

def expandRangeStartList(size, entries, key = lambda it: it[0], value = lambda it: it[1]):
  sorted_entries = sorted(entries, key=key)
  items = list(repeat(None, size))
  def assignRange(start, stop, value):
    items[start:stop] = repeat(value, stop - start)
  for (a, b) in zipWithNext(sorted_entries):
    assignRange(key(a), key(b), value(a))
  last_item = sorted_entries[-1]
  assignRange(key(last_item), size, value(last_item))
  return items

class PatternType:
  def __init__(self, regex, transform = identity):
    self.regex, self.transform = regex, transform
  def __repr__(self): return f"PatternType({self.regex})"
  def __call__(self, text):
    groups = findall(self.regex, text)
    return list(map(self.transform, groups))


def snakeSplit(text): return text.strip().split("_")
def titleCased(texts, sep = " "): return sep.join(map(str.capitalize, texts))

def printAttributes(fmt = lambda k, v: f"[{titleCased(snakeSplit(k))}] {v}", sep = "\n", file = stderr, **kwargs):
  entries = [fmt(k, v) for (k, v) in kwargs.items()]
  print(sep.join(entries), file=file)

class Reducer(Generic[T, R]):
  def __init__(self): pass
  def accept(self, value: T): pass
  def finish(self) -> R: pass
  def reduce(self, xs: Iterable[T]) -> R:
    for x in xs:
      self.accept(x)
    return self.finish()

class EffectReducer(Reducer[T, R]):
  ''' `Reducer` defined using makeBase/onAccept '''
  def __init__(self):
    self._base = self._makeBase()
  @classmethod #< used in ctor
  def _makeBase(cls): pass
  def _onAccept(self, base, value): pass
  def accept(self, value):
    self._onAccept(self._base, value)
  def finish(self):
    return self._base

class AsNoOp(Reducer[Any, None]):
  def __init__(self, *args): pass

class AsDict(EffectReducer[Tuple[K, V], Dict[K, V]]):
  @classmethod
  def _makeBase(cls): return dict()
  def _onAccept(self, base, value):
    (k, v) = value
    base[k] = v
