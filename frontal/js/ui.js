// Utilidades de la página que usan los demás módulos: avisos, botones y creación de elementos.

import { mensajeError } from "./sesion.js";

export const $ = (id) => document.getElementById(id);

/** Crea un elemento. `props` admite atributos y propiedades (className, textContent, onclick,
 *  hidden...); los hijos pueden ser elementos o textos. Los textos van siempre como texto, nunca
 *  como HTML, así que un nombre de juego con "<" no puede romper la página. */
export function el(etiqueta, props = {}, ...hijos) {
  const nodo = document.createElement(etiqueta);
  for (const [clave, valor] of Object.entries(props)) {
    if (valor === undefined || valor === null || valor === false) continue;
    if (clave in nodo) nodo[clave] = valor;
    else nodo.setAttribute(clave, valor === true ? "" : valor);
  }
  for (const hijo of hijos.flat()) {
    if (hijo !== null && hijo !== undefined && hijo !== false) nodo.append(hijo);
  }
  return nodo;
}

// --------------------------------------------------------------------------- avisos
// Un solo aviso flotante para toda la web. Los de "hecho" (tipo "ok") se van solos; los errores
// se quedan hasta que se cierran, se pulsa Escape o empieza otra acción (avisar(null)).
const SEGUNDOS_OK = 6;
let ocultar;

export function avisar(texto, tipo = "error") {
  const aviso = $("aviso");
  clearTimeout(ocultar);
  $("aviso-texto").textContent = texto ?? "";
  aviso.className = `aviso ${tipo}`;
  aviso.hidden = !texto;
  if (texto && tipo === "ok") ocultar = setTimeout(() => { aviso.hidden = true; }, SEGUNDOS_OK * 1000);
}

export function avisarError(e) {
  // Los errores de Firebase traen code ("auth/..."); los de la API (ErrorApi), estado
  avisar(e?.code ? mensajeError(e) : e?.message ?? String(e));
}

$("aviso-cerrar").addEventListener("click", () => avisar(null));
document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape" && !$("aviso").hidden) avisar(null);
});

/** Ejecuta una acción con el botón desactivado, para que un doble clic no la repita. Los errores
 *  se muestran en el aviso. */
export async function conBoton(boton, accion) {
  boton.disabled = true;
  try {
    await accion();
  } catch (e) {
    avisarError(e);
  } finally {
    if (boton.isConnected) boton.disabled = false;
  }
}