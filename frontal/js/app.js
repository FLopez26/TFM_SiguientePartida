// Página principal: entrada (Google o invitado), cabecera de la cuenta y botones de la cuenta.
// El perfil (búsqueda, listas y "para mí") está en perfil.js.

import { api } from "./api.js";
import * as perfil from "./perfil.js";
import * as sesion from "./sesion.js";
import { $, avisar, avisarError, conBoton } from "./ui.js";

// Firebase solo admite localhost como dominio autorizado, no 127.0.0.1
if (location.hostname === "127.0.0.1") {
  location.replace(location.href.replace("127.0.0.1", "localhost"));
}

// --------------------------------------------------------------------------- pantallas
function mostrar(pantalla) {   // "cargando", "entrada" o "perfil"
  for (const id of ["cargando", "entrada", "perfil"]) $(id).hidden = id !== pantalla;
  $("cuenta").hidden = pantalla !== "perfil";
}

function pintarCuenta(cuenta) {
  $("nombre").textContent = cuenta.es_invitado ? "Invitado" : cuenta.nombre_visible ?? "Tu cuenta";
  $("boton-pasar-a-google").hidden = !cuenta.es_invitado;
}

/** Registra el inicio de sesión en la API, pinta la cuenta y carga sus juegos. */
async function cargarCuenta() {
  pintarCuenta(await api.post("/cuenta/sesion"));
  await perfil.cargar();
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

// --------------------------------------------------------------------------- entrar
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

// --------------------------------------------------------------------------- cuenta
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

$("boton-borrar-cuenta").addEventListener("click", (ev) =>
  conBoton(ev.currentTarget, async () => {
    if (!confirm("Se borrarán tu cuenta y todas tus listas de juegos. No se puede deshacer. " +
                 "¿Borrar la cuenta?")) return;
    // Primero se confirma la identidad (ventana de Google): si se cierra, no se borra nada
    await sesion.confirmarIdentidad();
    await api.borrar("/cuenta");
    await sesion.borrarUsuario();
    avisar("Tu cuenta y tus listas se han borrado.", "ok");
  }));