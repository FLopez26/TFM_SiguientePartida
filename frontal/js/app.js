// Página principal: entrada (Google o invitado) y cabecera de la cuenta.
// El perfil con las listas y el modo "para mí" se añaden encima de esto.

import { api } from "./api.js";
import * as sesion from "./sesion.js";

// Firebase solo admite localhost como dominio autorizado, no 127.0.0.1
if (location.hostname === "127.0.0.1") {
  location.replace(location.href.replace("127.0.0.1", "localhost"));
}

const $ = (id) => document.getElementById(id);

// --------------------------------------------------------------------------- avisos
function avisar(texto, tipo = "error") {
  const aviso = $("aviso");
  aviso.textContent = texto ?? "";
  aviso.className = tipo;
  aviso.hidden = !texto;
}

function avisarError(e) {
  // Los errores de Firebase traen code ("auth/..."); los de la API, estado
  avisar(e?.code ? sesion.mensajeError(e) : e?.message ?? String(e));
}

/** Ejecuta una acción con el botón desactivado, para que un doble clic no la repita. */
async function conBoton(boton, accion) {
  boton.disabled = true;
  try {
    await accion();
  } catch (e) {
    avisarError(e);
  } finally {
    boton.disabled = false;
  }
}

// --------------------------------------------------------------------------- pantallas
function mostrar(pantalla) {   // "cargando", "entrada" o "perfil"
  for (const id of ["cargando", "entrada", "perfil"]) $(id).hidden = id !== pantalla;
  $("cuenta").hidden = pantalla !== "perfil";
}

function pintarCuenta(cuenta) {
  $("nombre").textContent = cuenta.es_invitado ? "Invitado" : cuenta.nombre_visible ?? "Tu cuenta";
  $("boton-pasar-a-google").hidden = !cuenta.es_invitado;
  const { coleccion, deseo, descarte } = cuenta.juegos;
  $("resumen").textContent =
    `Colección: ${coleccion} · Lista de deseos: ${deseo} · Descartes: ${descarte}`;
}

/** Registra el inicio de sesión en la API y pinta la cuenta. */
async function cargarCuenta() {
  pintarCuenta(await api.post("/cuenta/sesion"));
  mostrar("perfil");
}

// --------------------------------------------------------------------------- arranque
/** Comprueba la API y si las cuentas están configuradas; /salud responde 503 si falta MySQL o
 *  el modelo, pero también trae el JSON con lo que falla. */
async function comprobarApi() {
  try {
    const respuesta = await fetch("/api/salud");
    const salud = await respuesta.json();
    if (salud.cuentas?.estado !== "ok") {
      avisar("La API no tiene configuradas las cuentas: falta FIREBASE_PROJECT_ID en .env.");
    } else if (!respuesta.ok) {
      avisar("La API está arrancada, pero falla MySQL o el modelo: mira /api/salud.");
    }
  } catch {
    avisar("La API no responde: ¿está arrancado el contenedor api?");
  }
}

if (!sesion.configurada) {
  avisar("El inicio de sesión no está configurado: faltan FIREBASE_API_KEY o FIREBASE_PROJECT_ID en .env.");
  $("boton-google").disabled = $("boton-invitado").disabled = true;
}
comprobarApi();

sesion.alCambiarSesion(async (usuario) => {
  if (!usuario) {
    mostrar("entrada");
    return;
  }
  try {
    await cargarCuenta();
  } catch (e) {
    avisarError(e);
    mostrar("entrada");
  }
});

// --------------------------------------------------------------------------- botones
// Al entrar no hace falta llamar a la API aquí: lo hace alCambiarSesion
$("boton-google").addEventListener("click", (ev) =>
  conBoton(ev.currentTarget, async () => {
    avisar(null);
    await sesion.entrarConGoogle();
  }));

$("boton-invitado").addEventListener("click", (ev) =>
  conBoton(ev.currentTarget, async () => {
    avisar(null);
    await sesion.entrarComoInvitado();
  }));

$("boton-pasar-a-google").addEventListener("click", (ev) =>
  conBoton(ev.currentTarget, async () => {
    avisar(null);
    const tokenInvitado = await sesion.pasarAGoogle();
    if (tokenInvitado) {
      // La cuenta de Google ya existía: se traen los juegos del invitado
      const r = await api.post("/cuenta/fusionar", { token_invitado: tokenInvitado });
      avisar(`Ya estás en tu cuenta de Google. Se han traído ${r.movidos} juegos del invitado` +
             (r.ya_estaban ? ` (${r.ya_estaban} ya los tenías y se queda lo que tenías).` : "."),
             "ok");
    } else {
      avisar("Listo: tu colección ya está guardada en tu cuenta de Google.", "ok");
    }
    await cargarCuenta();
  }));

$("boton-salir").addEventListener("click", (ev) =>
  conBoton(ev.currentTarget, async () => {
    const usuario = sesion.usuarioActual();
    if (usuario?.isAnonymous) {
      // Un invitado no puede volver a entrar en la misma cuenta: sus datos se quedarían
      // huérfanos, así que se borran (antes se le avisa)
      if (!confirm("Si sales como invitado se borrará tu colección, porque no hay forma de " +
                   "volver a esta cuenta. Para conservarla, usa antes «Guardar con Google». " +
                   "¿Salir de todos modos?")) return;
      await api.borrar("/cuenta");
      await sesion.borrarUsuario();
    } else {
      await sesion.salir();
    }
    avisar(null);
  }));