// Llamadas a la API. Van a /api/..., en la misma dirección que la web: nginx las pasa al
// contenedor api. Con sesión, cada llamada lleva el ID token de Firebase.

import { tokenActual } from "./sesion.js";

export class ErrorApi extends Error {
  constructor(estado, mensaje) {
    super(mensaje);
    this.estado = estado;   // código HTTP; 0 si no hay conexión
  }
}

function mensajeDe(estado, datos) {
  const detalle = datos?.detail;
  if (typeof detalle === "string") return detalle;
  if (Array.isArray(detalle)) return detalle.map((e) => e.msg).join("; ");   // 422 de FastAPI
  if (estado === 502 || estado === 504) return "La API no responde: ¿está arrancado el contenedor api?";
  return `Error ${estado} de la API`;
}

async function llamar(metodo, ruta, cuerpo, { conSesion = true, renovar = false } = {}) {
  const cabeceras = {};
  if (cuerpo !== undefined) cabeceras["Content-Type"] = "application/json";
  if (conSesion) {
    const token = await tokenActual(renovar);
    if (!token) throw new ErrorApi(401, "No has iniciado sesión.");
    cabeceras.Authorization = `Bearer ${token}`;
  }
  let respuesta;
  try {
    respuesta = await fetch(`/api${ruta}`, {
      method: metodo,
      headers: cabeceras,
      body: cuerpo === undefined ? undefined : JSON.stringify(cuerpo),
    });
  } catch {
    throw new ErrorApi(0, "No hay conexión con la web: ¿está arrancado el contenedor frontal?");
  }
  // Un 401 puede ser un token que acaba de caducar: se pide uno nuevo y se repite una vez
  if (respuesta.status === 401 && conSesion && !renovar) {
    return llamar(metodo, ruta, cuerpo, { conSesion, renovar: true });
  }
  if (respuesta.status === 204) return null;
  const datos = await respuesta.json().catch(() => null);
  if (!respuesta.ok) throw new ErrorApi(respuesta.status, mensajeDe(respuesta.status, datos));
  return datos;
}

// Rutas con sesión por defecto; para las públicas (catálogo, /salud): { conSesion: false }
export const api = {
  get: (ruta, opciones) => llamar("GET", ruta, undefined, opciones),
  post: (ruta, cuerpo, opciones) => llamar("POST", ruta, cuerpo, opciones),
  put: (ruta, cuerpo, opciones) => llamar("PUT", ruta, cuerpo, opciones),
  borrar: (ruta, opciones) => llamar("DELETE", ruta, undefined, opciones),
};