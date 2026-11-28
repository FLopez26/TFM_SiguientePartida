// Navegación entre las pantallas de la web con la parte de la dirección que va detrás de "#":
// #/inicio, #/mis-juegos, #/juego/13... Así funcionan los botones de atrás y adelante del
// navegador y se puede guardar un marcador, sin recargar la página ni tocar nginx.
//
// Cada pantalla es un elemento con data-vista en index.html, y cada enlace del menú lleva
// data-ir con el nombre de su pantalla. Un módulo que tenga que cargar algo al abrir su pantalla
// lo registra con alEntrar(); lo que va detrás del nombre (el 13 de #/juego/13) le llega como
// parámetros.

import { avisarError } from "./ui.js";

const INICIO = "inicio";
const NOMBRE_WEB = "Siguiente Partida";

const alEntrarEn = new Map();   // pantalla -> función(parámetros)
let activa = false;             // sin sesión se ve la entrada y la dirección no se toca
let primera = true;

/** `funcion(parametros)` se llama cada vez que se abre la pantalla `vista`. */
export function alEntrar(vista, funcion) {
  alEntrarEn.set(vista, funcion);
}

/** Abre una pantalla desde el código: ir("juego", 13) lleva a #/juego/13. */
export function ir(vista, ...parametros) {
  location.hash = enlace(vista, ...parametros);
}

/** Dirección de una pantalla, para los href: enlace("juego", 13) da "#/juego/13". */
export function enlace(vista, ...parametros) {
  return "#/" + [vista, ...parametros].map((p) => encodeURIComponent(p)).join("/");
}

/** Lo llama app.js: con sesión se ve la pantalla de la dirección; sin sesión, la entrada. */
export function activar(si) {
  activa = si;
  if (si) mostrar();
}

function leerDireccion() {
  try {
    const partes = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean)
      .map(decodeURIComponent);
    return { vista: partes[0] ?? INICIO, parametros: partes.slice(1) };
  } catch {   // una dirección mal escrita a mano (un % suelto)
    return { vista: INICIO, parametros: [] };
  }
}

function mostrar() {
  if (!activa) return;
  let { vista, parametros } = leerDireccion();
  let destino = document.querySelector(`[data-vista="${CSS.escape(vista)}"]`);
  if (!destino) {   // una pantalla que no existe: al inicio, sin dejarla en el historial
    vista = INICIO;
    parametros = [];
    destino = document.querySelector(`[data-vista="${INICIO}"]`);
    history.replaceState(null, "", enlace(INICIO));
  }

  for (const pantalla of document.querySelectorAll("[data-vista]")) {
    pantalla.hidden = pantalla !== destino;
  }
  for (const a of document.querySelectorAll("[data-ir]")) {
    if (a.dataset.ir === vista) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  }
  // En pantallas estrechas el menú es una tira que se desliza: que se vea la pantalla abierta
  document.querySelector(`.menu [data-ir="${CSS.escape(vista)}"]`)
    ?.scrollIntoView({ block: "nearest", inline: "nearest" });

  // El título de la pestaña: data-titulo si lo tiene (el h1 del inicio es el saludo) o el h1
  const titulo = destino.querySelector("h1");
  const nombre = destino.dataset.titulo ?? titulo?.textContent;
  document.title = nombre ? `${nombre} | ${NOMBRE_WEB}` : NOMBRE_WEB;
  if (!primera) {
    // Al cambiar de pantalla se empieza arriba, y el lector de pantalla lee el título nuevo
    window.scrollTo(0, 0);
    titulo?.focus({ preventScroll: true });
  }
  primera = false;

  const funcion = alEntrarEn.get(vista);
  if (funcion) Promise.resolve().then(() => funcion(parametros)).catch(avisarError);
}

addEventListener("hashchange", mostrar);