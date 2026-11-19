-- Esquema app: datos de la aplicación (cuentas y listas de juegos de cada cuenta).
-- No se regenera con el pipeline: aquí están los datos que meten los usuarios.
-- Se puede ejecutar varias veces sin borrar nada.

CREATE TABLE IF NOT EXISTS app.cuenta (
    id_cuenta       INT AUTO_INCREMENT PRIMARY KEY,
    -- Identificador de Firebase Auth (Google o invitado anónimo). Distingue mayúsculas:
    -- con la intercalación por defecto (ai_ci) dos uid distintos podrían chocar.
    firebase_uid    VARCHAR(128) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    nombre_visible  VARCHAR(100) NULL COMMENT 'Solo con Google; NULL en invitados',
    es_invitado     BOOLEAN NOT NULL DEFAULT TRUE,
    creada_en       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ultimo_acceso   DATETIME NULL,
    CONSTRAINT uq_cuenta_uid UNIQUE (firebase_uid)
) COMMENT = 'Cuentas de la app. De Google solo se guarda el uid y el nombre visible';

-- Paso de invitado a Google:
--  * si Firebase enlaza la credencial (mismo uid): UPDATE es_invitado = FALSE, nombre_visible = ...
--  * si esa cuenta de Google ya existía (otro uid): se mueven las filas de cuenta_juego
--    del invitado a la cuenta existente y se borra la del invitado.

CREATE TABLE IF NOT EXISTS app.cuenta_juego (
    id_cuenta       INT NOT NULL,
    bgg_id          INT NOT NULL COMMENT 'gold.juego.bgg_id, sin clave foránea porque gold se regenera',
    lista           ENUM('coleccion', 'deseo', 'descarte') NOT NULL,
    nota            DECIMAL(3,1) NULL COMMENT 'Nota propia opcional, solo para juegos de la colección',
    actualizado_en  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    -- Una fila por cuenta y juego: un juego está en una sola lista a la vez
    PRIMARY KEY (id_cuenta, bgg_id),
    KEY ix_cuenta_juego_lista (id_cuenta, lista),
    CONSTRAINT fk_cuenta_juego_cuenta FOREIGN KEY (id_cuenta)
        REFERENCES app.cuenta (id_cuenta) ON DELETE CASCADE,
    CONSTRAINT ck_cuenta_juego_nota CHECK (nota IS NULL OR (lista = 'coleccion' AND nota BETWEEN 1 AND 10))
) COMMENT = 'Colección, lista de deseos y descartes de cada cuenta';