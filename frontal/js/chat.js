// Chat con el asistente (POST /chat). La conversación solo vive aquí, en memoria: la API no la
// guarda, y se pierde al recargar la página o al cambiar de cuenta. En cada turno se mandan los
// últimos mensajes, que son los que lee Gemini.

import { api } from "./api.js";
import { botonesGuardar, ficha } from "./perfil.js";
import { $, el } from "./ui.js";

const ENVIADOS = 10;   // mensajes por turno (la API acepta hasta 20 y a Gemini le pasa 10)
const BIENVENIDA = "Cuéntame qué buscas: cuántos vais a jugar, cuánto tiempo, si lo queréis " +
  "sencillo o complejo, qué temática os gusta o qué juegos os encantan. Si es para regalar, dímelo.";

let conversacion = [];   // {rol: "usuario" | "asistente", texto}, sin la bienvenida
let hayChat = true;      // false si la API no tiene clave de Gemini
let generacion = 0;      // cambia al empezar otra conversación: la respuesta de la anterior se ignora

/** Conversación nueva (al entrar, al cambiar de cuenta o con el botón). */
export function reiniciar() {
  generacion += 1;
  conversacion = [];
  $("chat-mensajes").replaceChildren(burbuja("asistente", BIENVENIDA));
  $("chat-texto").value = "";
}

/** Lo llama app.js con lo que dice /salud: sin clave de Gemini, el chat se desactiva. */
export function disponible(si) {
  hayChat = si;
  $("chat-no-disponible").hidden = si;
  bloquear(false);
}

function bloquear(si) {
  $("chat-texto").disabled = $("chat-enviar").disabled = si || !hayChat;
  $("chat-nueva").disabled = si;
}

// --------------------------------------------------------------------------- piezas
function burbuja(clases, texto, ...extra) {
  return el("div", { className: `mensaje ${clases}` }, el("p", { className: "texto" }, texto),
            ...extra);
}

const decimal = (n) => String(n).replace(".", ",");

function rango(minimo, maximo, unidad) {
  if (minimo && maximo) return `${decimal(minimo)}-${decimal(maximo)}${unidad}`;
  if (minimo) return `desde ${decimal(minimo)}${unidad}`;
  if (maximo) return `hasta ${decimal(maximo)}${unidad}`;
  return null;
}

function textoEntendido(e) {
  const f = e.filtros;
  const partes = [];
  if (f.jugadores) partes.push(`${f.jugadores} ${f.jugadores === 1 ? "jugador" : "jugadores"}`);
  const duracion = rango(f.duracion_min, f.duracion_max, " min");
  if (duracion) partes.push(duracion);
  const peso = rango(f.peso_min, f.peso_max, "");
  if (peso) partes.push(`complejidad ${peso}`);
  partes.push(...e.etiquetas.map((t) => t.nombre));
  if (e.le_gustan.length) partes.push(`a partir de ${e.le_gustan.map((j) => j.nombre).join(", ")}`);
  if (e.no_le_gustan.length) partes.push(`sin ${e.no_le_gustan.map((j) => j.nombre).join(", ")}`);
  if (e.con_coleccion) partes.push("con tu colección");
  const lineas = [];
  if (partes.length) lineas.push(`He buscado: ${partes.join(" · ")}`);
  if (e.no_encontrados.length) lineas.push(`No están en el catálogo: ${e.no_encontrados.join(", ")}`);
  return lineas.join("\n");
}

/** Las fichas de las recomendaciones, con los datos de la API (no con lo que escribe Gemini). */
function fichas(respuesta) {
  const gustan = new Set(respuesta.entendido.le_gustan.map((j) => j.bgg_id));
  return el("ol", { className: "juegos" }, respuesta.recomendaciones.map((r) => el("li", {},
    ficha(r.juego),
    el("p", { className: "porque" },
       !r.porque ? "De los más populares que encajan"
       : gustan.has(r.porque.bgg_id) ? `Porque te gusta ${r.porque.nombre}`
       : `Porque tienes ${r.porque.nombre}`),
    botonesGuardar(r.juego.bgg_id))));
}

function anadir(nodo) {
  const caja = $("chat-mensajes");
  caja.append(nodo);
  caja.scrollTop = caja.scrollHeight;
}

// --------------------------------------------------------------------------- enviar
async function enviar() {
  const campo = $("chat-texto");
  const texto = campo.value.trim();
  if (!texto || campo.disabled) return;
  campo.value = "";
  for (const aviso of $("chat-mensajes").querySelectorAll(".mensaje.error")) aviso.remove();
  conversacion.push({ rol: "usuario", texto });
  const mio = burbuja("usuario", texto);
  const pensando = burbuja("asistente pensando", "Pensando…");
  anadir(mio);
  anadir(pensando);
  bloquear(true);
  const esta = generacion;
  try {
    const r = await api.post("/chat", { mensajes: conversacion.slice(-ENVIADOS) });
    if (esta !== generacion) return;   // se ha empezado otra conversación mientras tanto
    conversacion.push({ rol: "asistente", texto: r.texto });
    const entendido = r.recomendaciones.length ? textoEntendido(r.entendido) : "";
    const respuesta = burbuja("asistente", r.texto,
      entendido ? el("p", { className: "entendido" }, entendido) : null,
      r.recomendaciones.length ? fichas(r) : null);
    if (r.recomendaciones.length) respuesta.classList.add("con-fichas");
    pensando.replaceWith(respuesta);
  } catch (e) {
    if (esta !== generacion) return;
    // Sin respuesta (cuota de Gemini, red...): el mensaje sale de la conversación y vuelve al
    // cuadro de texto, para reenviarlo cuando se pueda
    conversacion.pop();
    mio.remove();
    pensando.replaceWith(el("p", { className: "mensaje error", role: "alert" },
                            e?.message ?? String(e)));
    if (!campo.value) campo.value = texto;
  } finally {
    bloquear(false);
    campo.focus();
  }
}

$("chat-form").addEventListener("submit", (ev) => {
  ev.preventDefault();
  enviar();
});

// Intro envía; Mayúsculas + Intro hace un salto de línea
$("chat-texto").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey && !ev.isComposing) {
    ev.preventDefault();
    $("chat-form").requestSubmit();
  }
});

$("chat-nueva").addEventListener("click", () => {
  reiniciar();
  $("chat-texto").focus();
});