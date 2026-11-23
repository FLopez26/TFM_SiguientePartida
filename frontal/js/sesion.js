// Inicio de sesión con Firebase Auth: con Google o como invitado (anónimo).
// Este módulo solo habla con Firebase; las llamadas a la API están en api.js y app.js.
// La configuración pública del proyecto la sirve nginx en /config.js, a partir de .env.

import { initializeApp } from "https://www.gstatic.com/firebasejs/13.4.0/firebase-app.js";
import {
  GoogleAuthProvider, deleteUser, getAuth, linkWithPopup, onAuthStateChanged,
  reauthenticateWithPopup, signInAnonymously, signInWithCredential, signInWithPopup, signOut,
  updateProfile,
} from "https://www.gstatic.com/firebasejs/13.4.0/firebase-auth.js";
import { firebaseConfig } from "/config.js";

export const configurada = Boolean(firebaseConfig.apiKey && firebaseConfig.projectId);
const auth = configurada ? getAuth(initializeApp(firebaseConfig)) : null;
if (auth) auth.languageCode = "es";   // la ventana de Google, en español

function proveedorGoogle() {
  const proveedor = new GoogleAuthProvider();
  proveedor.setCustomParameters({ prompt: "select_account" });   // deja elegir la cuenta
  return proveedor;
}

/** Llama a `funcion(usuario)` al cargar la página y cada vez que se entra o se sale.
 *  `usuario` es null si no hay sesión (o si Firebase no está configurado). */
export function alCambiarSesion(funcion) {
  if (!auth) {
    funcion(null);
    return;
  }
  onAuthStateChanged(auth, funcion);
}

export function usuarioActual() {
  return auth?.currentUser ?? null;
}

/** ID token para la cabecera Authorization. El SDK lo guarda y lo renueva solo antes de que
 *  caduque (dura 1 hora); con `renovar` se pide uno nuevo aunque no haya caducado. */
export async function tokenActual(renovar = false) {
  const usuario = usuarioActual();
  return usuario ? usuario.getIdToken(renovar) : null;
}

// Ventana emergente y no redirección: con las protecciones de privacidad de los navegadores
// actuales, la redirección falla cuando la web no está en el dominio de Firebase
export async function entrarConGoogle() {
  await signInWithPopup(auth, proveedorGoogle());
}

export async function entrarComoInvitado() {
  await signInAnonymously(auth);
}

/** Pasa el invitado actual a una cuenta de Google. Devuelve:
 *  - null si se ha enlazado: el uid no cambia y sus juegos siguen siendo suyos;
 *  - el token del invitado si esa cuenta de Google ya existía: ya se ha entrado con ella y hay
 *    que llamar a POST /cuenta/fusionar con ese token para traer los juegos del invitado. */
export async function pasarAGoogle() {
  const invitado = auth.currentUser;
  const tokenInvitado = await invitado.getIdToken(true);   // recién pedido: vale 1 hora
  try {
    const { user } = await linkWithPopup(invitado, proveedorGoogle());
    const google = user.providerData.find((p) => p.providerId === "google.com");
    if (!user.displayName && google?.displayName) {
      await updateProfile(user, { displayName: google.displayName });
    }
    await user.getIdToken(true);   // token nuevo: con la identidad de Google y el nombre
    return null;
  } catch (e) {
    if (e.code !== "auth/credential-already-in-use") throw e;
    // El invitado se queda en Firebase sin usar; sus juegos los mueve la API al fusionar
    await signInWithCredential(auth, GoogleAuthProvider.credentialFromError(e));
    return tokenInvitado;
  }
}

export async function salir() {
  await signOut(auth);
}

/** Antes de borrar una cuenta de Google: vuelve a pedir la cuenta en la ventana de Google. Así
 *  Firebase deja borrar el usuario (pide haber entrado hace poco) y se confirma que es la persona.
 *  Si cierra la ventana, lanza el error y no se borra nada. Los invitados no la necesitan. */
export async function confirmarIdentidad() {
  const usuario = auth.currentUser;
  if (usuario && !usuario.isAnonymous) await reauthenticateWithPopup(usuario, proveedorGoogle());
}

/** Borra el usuario de Firebase (sus datos de MySQL los borra antes DELETE /cuenta). Si Firebase
 *  no deja (por ejemplo, porque pide haber entrado hace poco), al menos se cierra la sesión. */
export async function borrarUsuario() {
  try {
    await deleteUser(auth.currentUser);
  } catch {
    await signOut(auth);
  }
}

/** Texto para el usuario a partir de un error de Firebase; null si no hay que avisar de nada
 *  (por ejemplo, si ha cerrado la ventana de Google). */
export function mensajeError(e) {
  switch (e?.code) {
    case "auth/popup-closed-by-user":
    case "auth/cancelled-popup-request":
    case "auth/user-cancelled":
      return null;
    case "auth/popup-blocked":
      return "El navegador ha bloqueado la ventana de Google: permite las ventanas emergentes de localhost.";
    case "auth/unauthorized-domain":
      return "Firebase no admite esta dirección: abre la web en http://localhost:8080.";
    case "auth/operation-not-allowed":
    case "auth/admin-restricted-operation":
      return "Ese inicio de sesión no está activado en Firebase (Authentication > Método de inicio de sesión).";
    case "auth/network-request-failed":
      return "No hay conexión con Firebase.";
    case "auth/invalid-api-key":
    case "auth/api-key-not-valid.-please-pass-a-valid-api-key.":
      return "La FIREBASE_API_KEY de .env no es válida.";
    default:
      return e?.message ?? String(e);
  }
}