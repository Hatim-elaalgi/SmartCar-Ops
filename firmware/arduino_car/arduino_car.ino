// ============================================================
//              VOITURE AUTONOME
//       SUIVI DE LIGNE + EVITEMENT OBSTACLE
// ============================================================


// ======================= PINS ================================

// L298N
#define IN1 8
#define IN2 9
#define IN3 10
#define IN4 11

// Capteurs IR
#define IR_LEFT  2
#define IR_RIGHT 3

// HC-SR04
#define TRIG_PIN 12
#define ECHO_PIN 13


// ===================== PARAMETRES =============================

// Vitesse normale
const int MOTOR_SPEED = 190;

// Distance de détection
const int OBSTACLE_DISTANCE = 13;


// ------------------------------------------------------------
// TEMPS DES 3 ROTATIONS
// ------------------------------------------------------------

const int turn_1_time = 510;   // 90° gauche
const int turn_2_time = 350;   // 90° droite
const int turn_3_time = 330;   // 90° droite


// ------------------------------------------------------------
// TEMPS DES 2 AVANCÉES
// ------------------------------------------------------------

const int forward_1_time = 700;
const int forward_2_time = 600;


// ============================================================
// SETUP
// ============================================================

void setup()
{
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  pinMode(IR_LEFT, INPUT);
  pinMode(IR_RIGHT, INPUT);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  stopMotors();

  delay(1000);
}


// ============================================================
// LOOP
// ============================================================

void loop()
{
  long distance = getDistance();

  // ----------------------------------------------------------
  // OBSTACLE DETECTE
  // ----------------------------------------------------------

  if (distance <= OBSTACLE_DISTANCE)
  {
    avoidObstacle();
  }

  // ----------------------------------------------------------
  // PAS D'OBSTACLE
  // ----------------------------------------------------------

  else
  {
    followLine();
  }
}


// ============================================================
// SUIVI DE LIGNE
// ============================================================
//
// 0 0 = sol
// 1 1 = noir
//
// 0 1 = gauche sol / droite noir → DROITE
// 1 0 = gauche noir / droite sol → GAUCHE
//
// ============================================================

void followLine()
{
  int left = digitalRead(IR_LEFT);
  int right = digitalRead(IR_RIGHT);


  // ----------------------------------------------------------
  // SOL / SOL
  // ----------------------------------------------------------

  if (left == 0 && right == 0)
  {
    forward();
  }


  // ----------------------------------------------------------
  // NOIR / NOIR
  // ----------------------------------------------------------

  else if (left == 1 && right == 1)
  {
    forward();
  }


  // ----------------------------------------------------------
  // GAUCHE = SOL
  // DROITE = NOIR
  // → DROITE
  // ----------------------------------------------------------

  else if (left == 0 && right == 1)
  {
    turnRightLine();
  }


  // ----------------------------------------------------------
  // GAUCHE = NOIR
  // DROITE = SOL
  // → GAUCHE
  // ----------------------------------------------------------

  else if (left == 1 && right == 0)
  {
    turnLeftLine();
  }
}


// ============================================================
// AVANCER
// ============================================================

void forward()
{
  // Moteur gauche AVANT
  digitalWrite(IN1, LOW);
  analogWrite(IN2, MOTOR_SPEED);

  // Moteur droit AVANT
  digitalWrite(IN3, LOW);
  analogWrite(IN4, MOTOR_SPEED);
}


// ============================================================
// VIRAGE DROITE
// ============================================================

void turnRightLine()
{
  // Moteur gauche AVANT
  digitalWrite(IN1, LOW);
  analogWrite(IN2, MOTOR_SPEED);

  // Moteur droit STOP
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}


// ============================================================
// VIRAGE GAUCHE
// ============================================================

void turnLeftLine()
{
  // Moteur gauche STOP
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);

  // Moteur droit AVANT
  digitalWrite(IN3, LOW);
  analogWrite(IN4, MOTOR_SPEED);
}


// ============================================================
// ROTATION GAUCHE
// ============================================================
//
// Un seul moteur fonctionne afin d'éviter le problème
// de démarrage de la rotation sur place.
//
// ============================================================

void rotateLeft()
{
  // Moteur gauche STOP
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);

  // Moteur droit AVANT
  digitalWrite(IN3, LOW);
  analogWrite(IN4, MOTOR_SPEED);
}


// ============================================================
// ROTATION DROITE
// ============================================================

void rotateRight()
{
  // Moteur gauche AVANT
  digitalWrite(IN1, LOW);
  analogWrite(IN2, MOTOR_SPEED);

  // Moteur droit STOP
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}


// ============================================================
// STOP
// ============================================================

void stopMotors()
{
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);

  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}


// ============================================================
// HC-SR04
// ============================================================

long getDistance()
{
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);

  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);

  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000);

  if (duration == 0)
  {
    return 999;
  }

  return duration * 0.0343 / 2;
}


// ============================================================
// EVITEMENT D'OBSTACLE
// ============================================================
//
// NOUVELLE STRATEGIE:
//
// Obstacle détecté
//        ↓
// Tourner 90° GAUCHE
//        ↓
// Avancer
//        ↓
// Tourner 90° DROITE
//        ↓
// Avancer
//        ↓
// Tourner 90° DROITE
//        ↓
// Avancer jusqu'à retrouver la ligne
//
// ============================================================

void avoidObstacle()
{
  // ==========================================================
  // 1. TOURNER 90° À GAUCHE
  // ==========================================================

  rotateLeft();

  delay(turn_1_time);


  // ==========================================================
  // 2. AVANCER
  // ==========================================================

  forward();

  delay(forward_1_time);


  // ==========================================================
  // 3. TOURNER 90° À DROITE
  // ==========================================================

  rotateRight();

  delay(turn_2_time);


  // ==========================================================
  // 4. AVANCER
  // ==========================================================

  forward();

  delay(forward_2_time);


  // ==========================================================
  // 5. TOURNER 90° À DROITE
  // ==========================================================

  rotateRight();

  delay(turn_3_time);


  // ==========================================================
  // 6. AVANCER JUSQU'À RETROUVER LA LIGNE
  // ==========================================================

  findLine();
}


// ============================================================
// RETOUR AU RUBAN NOIR
// ============================================================

void findLine()
{
  // ----------------------------------------------------------
  // Avancer jusqu'à retrouver le noir
  // ----------------------------------------------------------

  forward();


  while (true)
  {
    int left = digitalRead(IR_LEFT);
    int right = digitalRead(IR_RIGHT);


    // --------------------------------------------------------
    // Un des deux capteurs détecte le noir
    // --------------------------------------------------------

    if (left == 1 || right == 1)
    {
      stopMotors();

      delay(200);

      return;
    }


    delay(20);
  }
}
