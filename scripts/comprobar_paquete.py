"""Comprueba, antes de subir a PyPI, que el paquete solo lleva la librería y su licencia.

Lee la versión de pyproject.toml y revisa los dos ficheros de dist/: el sdist (.tar.gz) y el
wheel (.whl). Si aparece cualquier otra cosa (.env, datos, tests, ml/, pipeline/...), la lista
y termina con error. Es la última comprobación antes de twine upload.

Uso (desde la raíz, después de python -m build):
    python scripts/comprobar_paquete.py
"""
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PAQUETE = "siguiente_partida"


def es_de_la_libreria(ruta: str) -> bool:
    """Código de la librería o su licencia."""
    return ruta.endswith(".py") or ruta.endswith("/LICENSE")


def no_esperados_sdist(nombres: list[str], base: str) -> list[str]:
    sueltos = {"", "MANIFEST.in", "PKG-INFO", "pyproject.toml", "setup.cfg", "src",
               f"src/{PAQUETE}", f"src/{PAQUETE}.egg-info"}
    raros = []
    for n in nombres:
        r = n.removeprefix(base).lstrip("/")
        if r in sueltos or r.startswith(f"src/{PAQUETE}.egg-info/"):
            continue
        if r.startswith(f"src/{PAQUETE}/") and es_de_la_libreria(r):
            continue
        raros.append(n)
    return raros


def no_esperados_wheel(nombres: list[str], base: str) -> list[str]:
    return [n for n in nombres
            if not n.startswith(f"{base}.dist-info/")
            and not (n.startswith(f"{PAQUETE}/") and es_de_la_libreria(n))]


def main() -> int:
    version = tomllib.loads((RAIZ / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    base = f"{PAQUETE}-{version}"
    sdist = RAIZ / "dist" / f"{base}.tar.gz"
    wheel = RAIZ / "dist" / f"{base}-py3-none-any.whl"
    for ruta in (sdist, wheel):
        if not ruta.exists():
            print(f"No existe {ruta.relative_to(RAIZ)}: ejecuta antes python -m build")
            return 1
    with tarfile.open(sdist) as t:
        en_sdist = t.getnames()
    with zipfile.ZipFile(wheel) as z:
        en_wheel = z.namelist()

    modulos = sorted(Path(n).name for n in en_wheel if n.startswith(f"{PAQUETE}/") and n.endswith(".py"))
    print(f"Versión {version}. Módulos en el wheel: {', '.join(modulos)}")
    raros = no_esperados_sdist(en_sdist, base) + no_esperados_wheel(en_wheel, base)
    if raros:
        print("NO ESPERADO, no subas nada:")
        print("\n".join(f"  {n}" for n in raros))
        return 1
    print(f"OK: el sdist ({len(en_sdist)} entradas) y el wheel ({len(en_wheel)}) solo llevan la "
          "librería y su licencia")
    return 0


if __name__ == "__main__":
    sys.exit(main())