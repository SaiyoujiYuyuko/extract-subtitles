from re import findall
from itertools import chain, islice
from sys import stderr

# == Normal function utils ==
identity = lambda x: x
noOp = lambda x: ()

def let(transform, x):
  return transform(x) if x != None else None

def also(op, x):
  op(x)
  return x

def require(p, message):
  if not p: raise ValueError(message)

def toMapper(transform):
  return lambda xs: [transform(x) for x in xs]

def zipWithNext(xs: list):
  require(len(xs) >= 2, f"len {len(xs)} is too short (< 2)")
  for i in range(1, len(xs)):
    yield (xs[i-1], xs[i])

def chunked(n, xs):
  while True:
    try: #< must return when inner gen finished
      first = next(xs)
    except StopIteration: return
    chunk = islice(xs, n)
    yield chain((first,) , chunk)

def collect2(selector2, xs):
  bs, cs = [], []
  for x in xs:
    b, c = selector2(x)
    bs.append(b); cs.append(c)
  return (bs, cs)

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

class Reducer:
  def __init__(self): pass
  def accept(self, value): pass
  def finish(self): pass
  def reduce(self, xs):
    for x in xs:
      self.accept(x)
    return self.finish()

class EffectReducer(Reducer):
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

class AsNoOp(Reducer):
  def __init__(self, *args): pass

class AsDict(EffectReducer):
  @classmethod
  def _makeBase(cls): return dict()
  def _onAccept(self, base, value):
    (k, v) = value
    base[k] = v
