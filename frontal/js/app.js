// Arranque de la web: las tres pantallas generales (cargando, entrada y la web con su menú) y la
// cuenta (la cabecera, el inicio y "Mi cuenta"). Cada parte del menú tiene su módulo: perfil.js
// (Mis juegos y Para mí) y chat.js (Asistente). La navegación entre pantallas está en rutas.js.

import { api } from "./api.js";
import * as chat from "./chat.js";
import * as perfil from "./perfil.js";
import * as rutas from "./rutas.js";
import * as sesion from "./sesion.js";
import { $, avisar, avisarError, conBoton } from "./ui.js";

// Firebase solo admite localhost como dominio autorizado, no 127.0.0.1
if (location.hostname === "127.0.0.1") {
  location.replace(location.href.replace("127.0.0.1", "localhost"));
}

// --------------------------------------------------------------------------- pantallas
function mostrar(pantalla) {   // "cargando", "entrada" o "web"
  for (const id of ["cargando", "entrada", "web"]) $(id).hidden = id !== pantalla;
  rutas.activar(pantalla === "web");
}

/** Al salir o borrar la cuenta, la próxima vez se entra por el inicio. */
function olvidarDireccion() {
  history.replaceState(null, "", location.pathname + location.search);
}

// --------------------------------------------------------------------------- cuenta
const cuantos = (n, uno, varios) => `${n} ${n === 1 ? uno : varios}`;

function pintarCuenta(cuenta) {
  const nombre = cuenta.es_invitado ? "Invitado" : cuenta.nombre_visible?.trim() || "Tu cuenta";
  $("chip-nombre").textContent = nombre;
  $("chip-inicial").textContent = nombre.charAt(0).toUpperCase();

  const nombrePila = cuenta.es_invitado ? null : cuenta.nombre_visible?.trim().split(/\s+/)[0];
  $("saludo").textContent = nombrePila ? `Hola, ${nombrePila}` : "Hola";
  $("cuenta-descripcion").textContent = cuenta.es_invitado
    ? "Has entrado como invitado. Tu colección está ligada a este navegador."
    : `Has entrado con tu cuenta de Google${cuenta.nombre_visible ? ` como ${cuenta.nombre_visible}` : ""}.`;
  $("cuenta-salir").textContent = cuenta.es_invitado
    ? "Como invitado, al salir se borra tu colección, porque no hay forma de volver a esta cuenta."
    : "Tus listas se quedan guardadas en tu cuenta de Google para la próxima vez.";
  for (const nodo of document.querySelectorAll(".solo-invitado")) nodo.hidden = !cuenta.es_invitado;
  pintarResumen(cuenta.juegos);
}

/** El resumen de las listas del inicio. */
function pintarResumen({ coleccion, deseo, descarte }) {
  $("resumen-listas").textContent = coleccion === 0
    ? "Aún no has añadido juegos. Empieza por los que tienes en casa: con ellos se hacen tus recomendaciones."
    : `Tienes ${cuantos(coleccion, "juego", "juegos")} en tu colección, ` +
      `${cuantos(deseo, "juego", "juegos")} en tu lista de deseos y ` +
      `${cuantos(descarte, "descartado", "descartados")}.`;
  const boton = $("resumen-boton");
  boton.textContent = coleccion === 0 ? "Añadir mis juegos" : "Ver mis juegos";
  boton.classList.toggle("principal", coleccion === 0);
}

/** Registra el inicio de sesión en la API, pinta la cuenta y carga sus juegos. */
async function cargarCuenta() {
  pintarCuenta(await api.post("/cuenta/sesion"));
  await perfil.cargar();
  chat.reiniciar();
  mostrar("web");
}

// Al volver al inicio, las listas pueden haber cambiado
rutas.alEntrar("inicio", async () => pintarResumen((await api.get("/cuenta")).juegos));

// --------------------------------------------------------------------------- arranque
/** Comprueba la API y si las cuentas están configuradas; /salud responde 503 si falta MySQL o
 *  el modelo, pero también trae el JSON con lo que falla. */
async function comprobarApi() {
  try {
    const respuesta = await fetch("/api/salud");
    const salud = await respuesta.json();
    chat.disponible(salud.chat?.estado === "ok");
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

// --------------------------------------------------------------------------- mi cuenta
// Hay un botón "Guardar con Google" en el inicio y otro en "Mi cuenta" (solo para invitados)
async function pasarAGoogle() {
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
}
for (const boton of document.querySelectorAll(".pasar-a-google")) {
  boton.addEventListener("click", (ev) => conBoton(ev.currentTarget, pasarAGoogle));
}

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
    olvidarDireccion();
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
    olvidarDireccion();
    avisar("Tu cuenta y tus listas se han borrado.", "ok");
  }));