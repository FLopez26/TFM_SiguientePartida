// Perfil: buscar juegos y guardarlos, las tres listas (colección, deseos y descartes) y el modo
// "para mí". Lo guardado vive en MySQL; aquí se guarda una copia para pintar sin pedirla cada vez.

import { api } from "./api.js";
import { $, avisar, avisarError, conBoton, el } from "./ui.js";

const LISTAS = {
  coleccion: { pestana: "Colección", guardar: "Lo tengo", mover: "A la colección",
               estado: "En tu colección", vacia: "Aún no hay juegos en tu colección: búscalos arriba." },
  deseo: { pestana: "Lista de deseos", guardar: "Lo quiero", mover: "A deseos",
           estado: "En tu lista de deseos", vacia: "Tu lista de deseos está vacía." },
  descarte: { pestana: "Descartes", guardar: "No me interesa", mover: "Descartar",
              estado: "Descartado", vacia: "No has descartado ningún juego." },
};
const RECOMENDACIONES = 10;

let guardados = new Map();   // bgg_id -> JuegoGuardado (copia de lo que hay en la cuenta)
let listaActiva = "coleccion";
let resultados = [];         // última búsqueda
let recomendaciones = null;  // última respuesta de "para mí"
let numeroBusqueda = 0;      // para descartar respuestas de búsquedas que ya no son la última

/** Carga los juegos de la cuenta y pinta el perfil desde cero (al entrar o al cambiar de cuenta). */
export async function cargar() {
  const juegos = await api.get("/cuenta/juegos");
  guardados = new Map(juegos.map((j) => [j.bgg_id, j]));
  resultados = [];
  recomendaciones = null;
  $("texto-busqueda").value = "";
  pintarTodo();
}

// --------------------------------------------------------------------------- guardar y quitar
async function guardar(bggId, lista, nota = null) {
  const juego = await api.put(`/cuenta/juegos/${bggId}`, nota === null ? { lista } : { lista, nota });
  guardados.set(bggId, juego);
  pintarTodo();
}

async function quitar(bggId) {
  await api.borrar(`/cuenta/juegos/${bggId}`);
  guardados.delete(bggId);
  pintarTodo();
}

// --------------------------------------------------------------------------- piezas
function anio(a) {
  if (a === null || a === undefined) return null;
  return el("span", { className: "anio" }, a < 0 ? ` (${-a} a. C.)` : ` (${a})`);
}

/** Imagen, nombre (enlace a BGG) y año. `juego` es null si ya no está en el catálogo. */
export function ficha(juego, bggId) {
  if (!juego) {
    return el("div", { className: "ficha" }, el("div", { className: "sin-imagen" }),
              el("span", { className: "nombre" }, `Juego ${bggId}`),
              el("span", { className: "nota" }, " · ya no está en el catálogo"));
  }
  return el("div", { className: "ficha" },
    juego.imagen_url ? el("img", { src: juego.imagen_url, alt: "", loading: "lazy" })
                     : el("div", { className: "sin-imagen" }),
    el("a", { className: "nombre", href: `https://boardgamegeek.com/boardgame/${juego.bgg_id}`,
              target: "_blank", rel: "noopener" }, juego.nombre ?? `Juego ${juego.bgg_id}`),
    anio(juego.anio));
}

/** Un botón por lista, salvo `menos`. `texto` es "guardar" (buscar y recomendaciones) o "mover"
 *  (dentro de las listas). */
function botonesListas(bggId, menos, texto) {
  return Object.entries(LISTAS).filter(([lista]) => lista !== menos).map(([lista, t]) =>
    el("button", { type: "button", textContent: t[texto],
                   onclick: (ev) => conBoton(ev.currentTarget, () => guardar(bggId, lista)) }));
}

/** Botones para guardar un juego desde fuera del perfil (las fichas del chat). Al guardarlo, dicen
 *  en qué lista ha quedado y dejan moverlo a las otras. */
export function botonesGuardar(bggId) {
  const caja = el("div", { className: "acciones" });
  const pintar = () => {
    const guardado = guardados.get(bggId);
    caja.replaceChildren(
      guardado ? el("span", { className: "estado" }, LISTAS[guardado.lista].estado) : "",
      ...Object.entries(LISTAS).filter(([lista]) => lista !== guardado?.lista).map(([lista, t]) =>
        el("button", { type: "button", textContent: t.guardar,
                       onclick: (ev) => conBoton(ev.currentTarget, async () => {
                         await guardar(bggId, lista);
                         pintar();
                       }) })));
  };
  pintar();
  return caja;
}

function selectorNota(guardado) {
  const opciones = [el("option", { value: "", textContent: "Sin nota" })];
  for (let n = 1; n <= 10; n++) opciones.push(el("option", { value: String(n), textContent: String(n) }));
  const nota = guardado.nota;
  if (nota !== null && !Number.isInteger(nota)) {   // una nota con decimales puesta desde la API
    opciones.push(el("option", { value: String(nota), textContent: String(nota).replace(".", ",") }));
  }
  const selector = el("select", { title: "Tu nota, de 1 a 10" }, opciones);
  selector.value = nota === null ? "" : String(nota);
  selector.onchange = () => conBoton(selector, () =>
    guardar(guardado.bgg_id, "coleccion", selector.value === "" ? null : Number(selector.value)));
  return el("label", { className: "nota-juego" }, "Nota ", selector);
}

// --------------------------------------------------------------------------- pintar
function pintarTodo() {
  pintarPestanas();
  pintarLista();
  pintarResultados();
  pintarRecomendaciones();
}

function pintarPestanas() {
  const cuantos = { coleccion: 0, deseo: 0, descarte: 0 };
  for (const j of guardados.values()) cuantos[j.lista] += 1;
  $("pestanas").replaceChildren(...Object.entries(LISTAS).map(([lista, t]) =>
    el("button", { type: "button", className: "pestana", "aria-pressed": String(lista === listaActiva),
                   textContent: `${t.pestana} (${cuantos[lista]})`,
                   onclick: () => { listaActiva = lista; pintarPestanas(); pintarLista(); } })));
}

function pintarLista() {
  const juegos = [...guardados.values()].filter((j) => j.lista === listaActiva)
    .sort((a, b) => (a.juego?.nombre ?? "").localeCompare(b.juego?.nombre ?? "", "es"));
  $("lista-juegos").replaceChildren(...juegos.map((j) => el("li", {},
    ficha(j.juego, j.bgg_id),
    el("div", { className: "acciones" },
      listaActiva === "coleccion" ? selectorNota(j) : null,
      botonesListas(j.bgg_id, listaActiva, "mover"),
      el("button", { type: "button", className: "secundario", textContent: "Quitar",
                     onclick: (ev) => conBoton(ev.currentTarget, () => quitar(j.bgg_id)) })))));
  const vacia = $("lista-vacia");
  vacia.textContent = LISTAS[listaActiva].vacia;
  vacia.hidden = juegos.length > 0;
}

function pintarResultados() {
  $("resultados").replaceChildren(...resultados.map((juego) => {
    const guardado = guardados.get(juego.bgg_id);
    return el("li", {}, ficha(juego),
      el("div", { className: "acciones" },
        guardado ? el("span", { className: "estado" }, LISTAS[guardado.lista].estado) : null,
        botonesListas(juego.bgg_id, guardado?.lista, "guardar")));
  }));
}

function pintarRecomendaciones() {
  const lista = $("recomendaciones");
  if (!recomendaciones) {
    lista.replaceChildren();
    $("explicacion-para-mi").hidden = true;
    return;
  }
  // Las que se acaban de guardar desde aquí ("Lo quiero"...) dejan de salir
  const visibles = recomendaciones.recomendaciones.filter((r) => !guardados.has(r.juego.bgg_id));
  lista.replaceChildren(...visibles.map((r) => el("li", {},
    ficha(r.juego),
    el("p", { className: "porque" },
       r.porque ? `Porque tienes ${r.porque.nombre}` : "De los más populares en BGG"),
    el("div", { className: "acciones" }, botonesListas(r.juego.bgg_id, null, "guardar")))));

  const tieneColeccion = [...guardados.values()].some((j) => j.lista === "coleccion");
  const ignorados = recomendaciones.ignorados.length;
  const textos = [];
  if (!tieneColeccion) textos.push("Tu colección está vacía, así que salen los más populares. Añade los juegos que tienes para que sean tuyas.");
  if (ignorados) textos.push(`${ignorados} ${ignorados === 1 ? "juego" : "juegos"} de tu colección no tienen valoraciones en los datos y no cuentan.`);
  if (recomendaciones.recomendaciones.length && !visibles.length) textos.push("Ya has guardado todas: pide más.");
  const explicacion = $("explicacion-para-mi");
  explicacion.textContent = textos.join(" ");
  explicacion.hidden = textos.length === 0;
}

// --------------------------------------------------------------------------- buscar
async function buscar(texto) {
  const pedido = ++numeroBusqueda;
  texto = texto.trim();
  if (texto.length < 2) {
    resultados = [];
    $("sin-resultados").hidden = true;
    pintarResultados();
    return;
  }
  const juegos = await api.get(`/juegos?q=${encodeURIComponent(texto)}&limite=10`,
                               { conSesion: false });
  if (pedido !== numeroBusqueda) return;   // mientras tanto se ha escrito otra cosa
  resultados = juegos;
  $("sin-resultados").hidden = juegos.length > 0;
  pintarResultados();
}

let espera;
$("texto-busqueda").addEventListener("input", (ev) => {
  // Se busca al dejar de escribir un momento, no con cada letra
  clearTimeout(espera);
  espera = setTimeout(() => buscar(ev.target.value).catch(avisarError), 300);
});

$("busqueda").addEventListener("submit", (ev) => {
  ev.preventDefault();
  clearTimeout(espera);
  buscar($("texto-busqueda").value).catch(avisarError);
});

// --------------------------------------------------------------------------- para mí
$("boton-para-mi").addEventListener("click", (ev) =>
  conBoton(ev.currentTarget, async () => {
    avisar(null);
    recomendaciones = await api.post("/cuenta/recomendaciones", { k: RECOMENDACIONES });
    pintarRecomendaciones();
  }));