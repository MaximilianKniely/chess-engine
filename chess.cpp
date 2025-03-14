#include <iostream>
#include <vector>
#include <cstdint>
#include <random>
#include <cassert>
#include <string>
#include <unordered_map>
#include <sstream>

// Constants for bitboard representation
const uint64_t RANK_1 = 0xFF;
const uint64_t RANK_2 = 0xFF00;
const uint64_t RANK_7 = 0xFF000000000000;
const uint64_t RANK_8 = 0xFF00000000000000;
const uint64_t FILE_A = 0x0101010101010101;
const uint64_t FILE_H = 0x8080808080808080;

// Magic numbers for sliding piece move generation
struct MagicEntry {
    uint64_t mask;
    uint64_t magic;
    int shift;
    uint64_t* attacks;
};

class ChessEngine {
private:
    // Move structure to represent a move
    struct Move {
        int from;
        int to;
        int pieceType;      // 0=pawn, 1=knight, 2=bishop, 3=rook, 4=queen, 5=king
        int promotion;      // 0=none, 1=knight, 2=bishop, 3=rook, 4=queen
        bool capture;
        bool doublePawnPush;
        bool enPassant;
        bool castling;
        int capturedPieceType; // 0=pawn, 1=knight, 2=bishop, 3=rook, 4=queen, 5=king

        std::string toString() const;
    };

    struct MoveState {
        int enPassantSquare;
        int halfMoveClock;
        uint8_t castlingRights;
        int capturedPieceType;
    };
    std::vector<MoveState> stateHistory;

    // Bitboards for each piece type and color
    uint64_t whitePawns, whiteKnights, whiteBishops, whiteRooks, whiteQueens, whiteKing;
    uint64_t blackPawns, blackKnights, blackBishops, blackRooks, blackQueens, blackKing;

    // Board state
    bool whiteToMove;
    uint8_t castlingRights; // KQkq - 1111 binary
    int enPassantSquare;
    int halfMoveClock;
    int fullMoveNumber;

    // Magic bitboard tables
    MagicEntry bishopMagics[64];
    MagicEntry rookMagics[64];

    // Precomputed attack tables
    uint64_t pawnAttacks[2][64]; // [color][square]
    uint64_t knightAttacks[64];
    uint64_t kingAttacks[64];

    // Helper methods
    void initializeAttackTables();
    void initializeMagicBitboards();
    uint64_t generateRookAttacks(int square, uint64_t blockers);
    uint64_t generateBishopAttacks(int square, uint64_t blockers);
    uint64_t generatePawnAttacks(int square, bool isWhite);
    uint64_t generateKnightAttacks(int square);
    uint64_t generateKingAttacks(int square);

    // Get combined bitboards
    uint64_t getWhitePieces() const {
        return whitePawns | whiteKnights | whiteBishops | whiteRooks | whiteQueens | whiteKing;
    }

    uint64_t getBlackPieces() const {
        return blackPawns | blackKnights | blackBishops | blackRooks | blackQueens | blackKing;
    }

    uint64_t getAllPieces() const {
        return getWhitePieces() | getBlackPieces();
    }

    // Move generation helpers
    std::vector<Move> generatePawnMoves(bool isWhite);
    std::vector<Move> generateKnightMoves(bool isWhite);
    std::vector<Move> generateBishopMoves(bool isWhite);
    std::vector<Move> generateRookMoves(bool isWhite);
    std::vector<Move> generateQueenMoves(bool isWhite);
    std::vector<Move> generateKingMoves(bool isWhite);

    // Check detection
    bool isSquareAttacked(int square, bool byWhite);
    bool isInCheck(bool isWhite);

    // Random number generation for magic numbers
    uint64_t generateRandom64();

public:


    // Constructor sets up the initial board position
    ChessEngine();

    // Setup the board from FEN string
    void setPosition(const std::string& fen);

    // Get all legal moves for the current side to move
    std::vector<Move> getLegalMoves(bool isWhite);

    // Make a move on the board
    bool makeMove(const Move& move);

    // Unmake a move (take back)
    void unmakeMove(const Move& move);

    // Get FEN representation of current position
    std::string getFEN() const;

    // Print the board for debugging
    void printBoard() const;
};

// Bit manipulation helpers
inline int bitScanForward(uint64_t bb) {
    if (bb == 0) return -1;
    return __builtin_ctzll(bb);
}

inline int bitScanReverse(uint64_t bb) {
    if (bb == 0) return -1;
    return 63 - __builtin_clzll(bb);
}

inline int popCount(uint64_t bb) {
    return __builtin_popcountll(bb);
}

inline uint64_t lsb(uint64_t bb) {
    return bb & -bb; // Isolate least significant bit
}

inline uint64_t popLsb(uint64_t& bb) {
    uint64_t bit = lsb(bb);
    bb &= bb - 1; // Clear least significant bit
    return bit;
}

// Constructor implementation
ChessEngine::ChessEngine() {
    // Initialize to starting position
    setPosition("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    // Initialize attack tables
    initializeAttackTables();

    // Initialize magic bitboards
    initializeMagicBitboards();
}

// The main getLegalMoves function
std::vector<ChessEngine::Move> ChessEngine::getLegalMoves(bool isWhite) {
    std::vector<Move> pseudoLegalMoves;

    // Generate all pseudo-legal moves - these functions need to be implemented to return vectors
    auto pawnMoves = generatePawnMoves(isWhite);
    auto knightMoves = generateKnightMoves(isWhite);
    auto bishopMoves = generateBishopMoves(isWhite);
    auto rookMoves = generateRookMoves(isWhite);
    auto queenMoves = generateQueenMoves(isWhite);
    auto kingMoves = generateKingMoves(isWhite);

    // Combine all moves
    pseudoLegalMoves.insert(pseudoLegalMoves.end(), pawnMoves.begin(), pawnMoves.end());
    pseudoLegalMoves.insert(pseudoLegalMoves.end(), knightMoves.begin(), knightMoves.end());
    pseudoLegalMoves.insert(pseudoLegalMoves.end(), bishopMoves.begin(), bishopMoves.end());
    pseudoLegalMoves.insert(pseudoLegalMoves.end(), rookMoves.begin(), rookMoves.end());
    pseudoLegalMoves.insert(pseudoLegalMoves.end(), queenMoves.begin(), queenMoves.end());
    pseudoLegalMoves.insert(pseudoLegalMoves.end(), kingMoves.begin(), kingMoves.end());

    // Filter out illegal moves (that leave own king in check)
    std::vector<Move> legalMoves;
    for (const auto& move : pseudoLegalMoves) {
        // Make the move
        bool moveValid = makeMove(move);
        if (!moveValid) continue;

        // If king is not in check after move, it's legal
        if (!isInCheck(isWhite)) {
            legalMoves.push_back(move);
        }

        // Unmake the move
        unmakeMove(move);
    }

    return legalMoves;
}

// Magic bitboard initialization (placeholder - actual implementation would be more complex)
void ChessEngine::initializeMagicBitboards() {
    // This would normally involve precalculating magic numbers and attack tables
    // Simplified version for demonstration
    for (int square = 0; square < 64; square++) {
        rookMagics[square].magic = generateRandom64() & generateRandom64() & generateRandom64();
        rookMagics[square].shift = 64 - popCount(generateRookAttacks(square, 0));
        rookMagics[square].mask = generateRookAttacks(square, 0);
        rookMagics[square].attacks = new uint64_t[1 << (64 - rookMagics[square].shift)];

        bishopMagics[square].magic = generateRandom64() & generateRandom64() & generateRandom64();
        bishopMagics[square].shift = 64 - popCount(generateBishopAttacks(square, 0));
        bishopMagics[square].mask = generateBishopAttacks(square, 0);
        bishopMagics[square].attacks = new uint64_t[1 << (64 - bishopMagics[square].shift)];
    }

    // Fill the attack tables (simplified)
    // In a real implementation, this would involve more sophisticated algorithms
}

uint64_t ChessEngine::generateRandom64() {
    static std::random_device rd;
    static std::mt19937_64 eng(rd());
    static std::uniform_int_distribution<uint64_t> distr;
    return distr(eng);
}

// Method to get sliding piece attacks using magic bitboards
uint64_t ChessEngine::generateRookAttacks(int square, uint64_t blockers) {
    // In a full implementation, this would use the magic bitboards
    // Simplified version for demonstration
    uint64_t attacks = 0;
    int rank = square / 8;
    int file = square % 8;

    // North
    for (int r = rank + 1; r <= 7; r++) {
        int sq = r * 8 + file;
        attacks |= (1ULL << sq);
        if (blockers & (1ULL << sq)) break;
    }

    // South
    for (int r = rank - 1; r >= 0; r--) {
        int sq = r * 8 + file;
        attacks |= (1ULL << sq);
        if (blockers & (1ULL << sq)) break;
    }

    // East
    for (int f = file + 1; f <= 7; f++) {
        int sq = rank * 8 + f;
        attacks |= (1ULL << sq);
        if (blockers & (1ULL << sq)) break;
    }

    // West
    for (int f = file - 1; f >= 0; f--) {
        int sq = rank * 8 + f;
        attacks |= (1ULL << sq);
        if (blockers & (1ULL << sq)) break;
    }

    return attacks;
}

uint64_t ChessEngine::generateBishopAttacks(int square, uint64_t blockers) {
    // Similar to rook attacks, but for diagonals
    uint64_t attacks = 0;
    int rank = square / 8;
    int file = square % 8;

    // North-East
    for (int r = rank + 1, f = file + 1; r <= 7 && f <= 7; r++, f++) {
        int sq = r * 8 + f;
        attacks |= (1ULL << sq);
        if (blockers & (1ULL << sq)) break;
    }

    // South-East
    for (int r = rank - 1, f = file + 1; r >= 0 && f <= 7; r--, f++) {
        int sq = r * 8 + f;
        attacks |= (1ULL << sq);
        if (blockers & (1ULL << sq)) break;
    }

    // South-West
    for (int r = rank - 1, f = file - 1; r >= 0 && f >= 0; r--, f--) {
        int sq = r * 8 + f;
        attacks |= (1ULL << sq);
        if (blockers & (1ULL << sq)) break;
    }

    // North-West
    for (int r = rank + 1, f = file - 1; r <= 7 && f >= 0; r++, f--) {
        int sq = r * 8 + f;
        attacks |= (1ULL << sq);
        if (blockers & (1ULL << sq)) break;
    }

    return attacks;
}

// Other methods would be similarly implemented to complete the engine

// Basic implementation of the missing move generation functions
std::vector<ChessEngine::Move> ChessEngine::generatePawnMoves(bool isWhite) {
    std::vector<Move> moves;
    uint64_t pawns = isWhite ? whitePawns : blackPawns;
    uint64_t enemies = isWhite ? getBlackPieces() : getWhitePieces();
    uint64_t empty = ~getAllPieces();

    // Direction multiplier (white moves up, black moves down)
    int dir = isWhite ? 1 : -1;
    int startRank = isWhite ? 1 : 6;
    int promotionRank = isWhite ? 7 : 0;

    // Process each pawn
    while (pawns) {
        int from = bitScanForward(pawns);
        uint64_t pawnBit = 1ULL << from;
        pawns &= ~pawnBit;

        int rank = from / 8;
        int file = from % 8;

        // Single push
        int to = from + 8 * dir;
        if (to >= 0 && to < 64 && (empty & (1ULL << to))) {
            // Check for promotion
            if (rank + dir == promotionRank) {
                // Add all promotion types
                for (int promo = 1; promo <= 4; promo++) {
                    Move move = {from, to, 0, promo, false, false, false, false};
                    moves.push_back(move);
                }
            } else {
                Move move = {from, to, 0, 0, false, false, false, false};
                moves.push_back(move);

                // Double push (only from starting position)
                if (rank == startRank) {
                    to = from + 16 * dir;
                    if (to >= 0 && to < 64 && (empty & (1ULL << to))) {
                        Move move = {from, to, 0, 0, false, true, false, false};
                        moves.push_back(move);
                    }
                }
            }
        }

        // Captures to the left
        if (file > 0) {
            to = from + 8 * dir - 1;
            if (to >= 0 && to < 64) {
                uint64_t targetBit = 1ULL << to;

                // Normal capture
                if (enemies & targetBit) {
                    if (rank + dir == promotionRank) {
                        // Add all promotion captures
                        for (int promo = 1; promo <= 4; promo++) {
                            Move move = {from, to, 0, promo, true, false, false, false};
                            moves.push_back(move);
                        }
                    } else {
                        Move move = {from, to, 0, 0, true, false, false, false};
                        moves.push_back(move);
                    }
                }

                // En passant capture
                if (to == enPassantSquare) {
                    Move move = {from, to, 0, 0, true, false, true, false};
                    moves.push_back(move);
                }
            }
        }

        // Captures to the right
        if (file < 7) {
            to = from + 8 * dir + 1;
            if (to >= 0 && to < 64) {
                uint64_t targetBit = 1ULL << to;

                // Normal capture
                if (enemies & targetBit) {
                    if (rank + dir == promotionRank) {
                        // Add all promotion captures
                        for (int promo = 1; promo <= 4; promo++) {
                            Move move = {from, to, 0, promo, true, false, false, false};
                            moves.push_back(move);
                        }
                    } else {
                        Move move = {from, to, 0, 0, true, false, false, false};
                        moves.push_back(move);
                    }
                }

                // En passant capture
                if (to == enPassantSquare) {
                    Move move = {from, to, 0, 0, true, false, true, false};
                    moves.push_back(move);
                }
            }
        }
    }

    return moves;
}

std::vector<ChessEngine::Move> ChessEngine::generateKnightMoves(bool isWhite) {
    std::vector<Move> moves;
    uint64_t knights = isWhite ? whiteKnights : blackKnights;
    uint64_t ownPieces = isWhite ? getWhitePieces() : getBlackPieces();

    // Knight movement offsets
    const int offsets[] = {-17, -15, -10, -6, 6, 10, 15, 17};

    while (knights) {
        int from = bitScanForward(knights);
        uint64_t knightBit = 1ULL << from;
        knights &= ~knightBit;

        int rank = from / 8;
        int file = from % 8;

        for (int offset : offsets) {
            int to = from + offset;

            // Check if the destination is valid
            if (to < 0 || to >= 64) continue;

            // Check if the knight move is valid (avoid wrapping around the board)
            int toRank = to / 8;
            int toFile = to % 8;
            int rankDiff = abs(toRank - rank);
            int fileDiff = abs(toFile - file);
            if (!((rankDiff == 2 && fileDiff == 1) || (rankDiff == 1 && fileDiff == 2))) continue;

            uint64_t targetBit = 1ULL << to;

            // Can't capture own pieces
            if (ownPieces & targetBit) continue;

            // Generate move (capture if enemy piece is present)
            bool isCapture = (getAllPieces() & targetBit) != 0;
            Move move = {from, to, 1, 0, isCapture, false, false, false};
            moves.push_back(move);
        }
    }

    return moves;
}

std::vector<ChessEngine::Move> ChessEngine::generateBishopMoves(bool isWhite) {
    std::vector<Move> moves;
    uint64_t bishops = isWhite ? whiteBishops : blackBishops;
    uint64_t ownPieces = isWhite ? getWhitePieces() : getBlackPieces();
    uint64_t allPieces = getAllPieces();

    // Process each bishop
    while (bishops) {
        int from = bitScanForward(bishops);
        uint64_t bishopBit = 1ULL << from;
        bishops &= ~bishopBit;

        // Generate attacks for this bishop
        uint64_t attacks = generateBishopAttacks(from, allPieces);

        // Remove attacks that hit our own pieces
        attacks &= ~ownPieces;

        // Process each attack
        while (attacks) {
            int to = bitScanForward(attacks);
            uint64_t targetBit = 1ULL << to;
            attacks &= ~targetBit;

            // Check if this is a capture
            bool isCapture = (allPieces & targetBit) != 0;

            Move move = {from, to, 2, 0, isCapture, false, false, false};
            moves.push_back(move);
        }
    }

    return moves;
}

std::vector<ChessEngine::Move> ChessEngine::generateRookMoves(bool isWhite) {
    std::vector<Move> moves;
    uint64_t rooks = isWhite ? whiteRooks : blackRooks;
    uint64_t ownPieces = isWhite ? getWhitePieces() : getBlackPieces();
    uint64_t allPieces = getAllPieces();

    // Process each rook
    while (rooks) {
        int from = bitScanForward(rooks);
        uint64_t rookBit = 1ULL << from;
        rooks &= ~rookBit;

        // Generate attacks for this rook
        uint64_t attacks = generateRookAttacks(from, allPieces);

        // Remove attacks that hit our own pieces
        attacks &= ~ownPieces;

        // Process each attack
        while (attacks) {
            int to = bitScanForward(attacks);
            uint64_t targetBit = 1ULL << to;
            attacks &= ~targetBit;

            // Check if this is a capture
            bool isCapture = (allPieces & targetBit) != 0;

            Move move = {from, to, 3, 0, isCapture, false, false, false};
            moves.push_back(move);
        }
    }

    return moves;
}

std::vector<ChessEngine::Move> ChessEngine::generateQueenMoves(bool isWhite) {
    std::vector<Move> moves;
    uint64_t queens = isWhite ? whiteQueens : blackQueens;
    uint64_t ownPieces = isWhite ? getWhitePieces() : getBlackPieces();
    uint64_t allPieces = getAllPieces();

    // Process each queen
    while (queens) {
        int from = bitScanForward(queens);
        uint64_t queenBit = 1ULL << from;
        queens &= ~queenBit;

        // Generate attacks for this queen (combine bishop and rook attacks)
        uint64_t attacks = generateBishopAttacks(from, allPieces) |
                           generateRookAttacks(from, allPieces);

        // Remove attacks that hit our own pieces
        attacks &= ~ownPieces;

        // Process each attack
        while (attacks) {
            int to = bitScanForward(attacks);
            uint64_t targetBit = 1ULL << to;
            attacks &= ~targetBit;

            // Check if this is a capture
            bool isCapture = (allPieces & targetBit) != 0;

            Move move = {from, to, 4, 0, isCapture, false, false, false};
            moves.push_back(move);
        }
    }

    return moves;
}

bool ChessEngine::isSquareAttacked(int square, bool byWhite) {
    uint64_t squareBit = 1ULL << square;

    // Check for pawn attacks
    uint64_t pawns = byWhite ? whitePawns : blackPawns;
    uint64_t pawnAttackMask = pawnAttacks[!byWhite][square]; // Use opposite color's attack perspective
    if (pawns & pawnAttackMask) return true;

    // Check for knight attacks
    uint64_t knights = byWhite ? whiteKnights : blackKnights;
    if (knights & knightAttacks[square]) return true;

    // Check for king attacks
    uint64_t king = byWhite ? whiteKing : blackKing;
    if (king & kingAttacks[square]) return true;

    // Check for sliding piece attacks (rooks, bishops, queens)
    uint64_t queens = byWhite ? whiteQueens : blackQueens;
    uint64_t bishops = byWhite ? whiteBishops : blackBishops;
    uint64_t rooks = byWhite ? whiteRooks : blackRooks;
    uint64_t blockers = getAllPieces();

    // Bishop-like attacks (diagonals)
    uint64_t bishopAttacks = generateBishopAttacks(square, blockers);
    if ((bishops | queens) & bishopAttacks) return true;

    // Rook-like attacks (horizontals/verticals)
    uint64_t rookAttacks = generateRookAttacks(square, blockers);
    if ((rooks | queens) & rookAttacks) return true;

    return false;
}

std::vector<ChessEngine::Move> ChessEngine::generateKingMoves(bool isWhite) {
    std::vector<Move> moves;
    uint64_t kingBB = isWhite ? whiteKing : blackKing;
    uint64_t ownPieces = isWhite ? getWhitePieces() : getBlackPieces();
    int from = bitScanForward(kingBB);

    // Skip if no king (shouldn't happen in a valid position)
    if (from == -1) return moves;

    // Regular king moves (all 8 directions)
    int offsets[] = {-9, -8, -7, -1, 1, 7, 8, 9};
    int rank = from / 8;
    int file = from % 8;

    for (int offset : offsets) {
        int to = from + offset;

        // Check if the destination is valid
        if (to < 0 || to >= 64) continue;

        // Check if the king move is valid (avoid wrapping around the board)
        int toRank = to / 8;
        int toFile = to % 8;
        int rankDiff = abs(toRank - rank);
        int fileDiff = abs(toFile - file);
        if (rankDiff > 1 || fileDiff > 1) continue;

        uint64_t targetBit = 1ULL << to;

        // Can't move to squares occupied by our pieces
        if (ownPieces & targetBit) continue;

        // Skip if the square is attacked (this is done in isInCheck after making the move)

        // Check if this is a capture
        bool isCapture = (getAllPieces() & targetBit) != 0;

        Move move = {from, to, 5, 0, isCapture, false, false, false};
        moves.push_back(move);
    }

    // Castling moves
    if (isWhite) {
        // White kingside castling
        if ((castlingRights & 1) &&
            !(getAllPieces() & 0x0000000000000060ULL) &&  // f1 and g1 are empty
            !isSquareAttacked(4, false) &&                // e1 not attacked
            !isSquareAttacked(5, false) &&                // f1 not attacked
            !isSquareAttacked(6, false)) {                // g1 not attacked
            Move move = {from, 6, 5, 0, false, false, false, true};
            moves.push_back(move);
        }

        // White queenside castling
        if ((castlingRights & 2) &&
            !(getAllPieces() & 0x000000000000000EULL) &&  // b1, c1, d1 are empty
            !isSquareAttacked(4, false) &&                // e1 not attacked
            !isSquareAttacked(3, false) &&                // d1 not attacked
            !isSquareAttacked(2, false)) {                // c1 not attacked
            Move move = {from, 2, 5, 0, false, false, false, true};
            moves.push_back(move);
        }
    } else {
        // Black kingside castling
        if ((castlingRights & 4) &&
            !(getAllPieces() & 0x6000000000000000ULL) &&  // f8 and g8 are empty
            !isSquareAttacked(60, true) &&                // e8 not attacked
            !isSquareAttacked(61, true) &&                // f8 not attacked
            !isSquareAttacked(62, true)) {                // g8 not attacked
            Move move = {from, 62, 5, 0, false, false, false, true};
            moves.push_back(move);
        }

        // Black queenside castling
        if ((castlingRights & 8) &&
            !(getAllPieces() & 0x0E00000000000000ULL) &&  // b8, c8, d8 are empty
            !isSquareAttacked(60, true) &&                // e8 not attacked
            !isSquareAttacked(59, true) &&                // d8 not attacked
            !isSquareAttacked(58, true)) {                // c8 not attacked
            Move move = {from, 58, 5, 0, false, false, false, true};
            moves.push_back(move);
        }
    }

    return moves;
}

void ChessEngine::initializeAttackTables() {
    // Initialize pawn attack tables
    for (int square = 0; square < 64; square++) {
        uint64_t bitboard = 1ULL << square;
        int file = square % 8;

        // White pawn attacks (up-left and up-right)
        uint64_t whiteAttacks = 0;
        if (file > 0) whiteAttacks |= (bitboard << 7);  // up-left
        if (file < 7) whiteAttacks |= (bitboard << 9);  // up-right
        pawnAttacks[1][square] = whiteAttacks;

        // Black pawn attacks (down-left and down-right)
        uint64_t blackAttacks = 0;
        if (file > 0) blackAttacks |= (bitboard >> 9);  // down-left
        if (file < 7) blackAttacks |= (bitboard >> 7);  // down-right
        pawnAttacks[0][square] = blackAttacks;
    }

    // Initialize knight attack table
    for (int square = 0; square < 64; square++) {
        int rank = square / 8;
        int file = square % 8;
        uint64_t attacks = 0;

        // All 8 knight moves
        if (rank > 1 && file > 0) attacks |= (1ULL << (square - 17));  // down 2, left 1
        if (rank > 1 && file < 7) attacks |= (1ULL << (square - 15));  // down 2, right 1
        if (rank > 0 && file > 1) attacks |= (1ULL << (square - 10));  // down 1, left 2
        if (rank > 0 && file < 6) attacks |= (1ULL << (square - 6));   // down 1, right 2
        if (rank < 7 && file > 1) attacks |= (1ULL << (square + 6));   // up 1, left 2
        if (rank < 7 && file < 6) attacks |= (1ULL << (square + 10));  // up 1, right 2
        if (rank < 6 && file > 0) attacks |= (1ULL << (square + 15));  // up 2, left 1
        if (rank < 6 && file < 7) attacks |= (1ULL << (square + 17));  // up 2, right 1

        knightAttacks[square] = attacks;
    }

    // Initialize king attack table
    for (int square = 0; square < 64; square++) {
        int rank = square / 8;
        int file = square % 8;
        uint64_t attacks = 0;

        // All 8 king moves
        if (rank > 0) attacks |= (1ULL << (square - 8));                // down
        if (rank < 7) attacks |= (1ULL << (square + 8));                // up
        if (file > 0) attacks |= (1ULL << (square - 1));                // left
        if (file < 7) attacks |= (1ULL << (square + 1));                // right
        if (rank > 0 && file > 0) attacks |= (1ULL << (square - 9));    // down-left
        if (rank > 0 && file < 7) attacks |= (1ULL << (square - 7));    // down-right
        if (rank < 7 && file > 0) attacks |= (1ULL << (square + 7));    // up-left
        if (rank < 7 && file < 7) attacks |= (1ULL << (square + 9));    // up-right

        kingAttacks[square] = attacks;
    }

    // The sliding piece attacks (bishop, rook, queen) are handled by the magic bitboard functions
    // and don't need to be precomputed the same way
}

void ChessEngine::setPosition(const std::string& fen) {
     // Reset all bitboards
     whitePawns = whiteKnights = whiteBishops = whiteRooks = whiteQueens = whiteKing = 0;
     blackPawns = blackKnights = blackBishops = blackRooks = blackQueens = blackKing = 0;

     // Reset game state
     castlingRights = 0;
     enPassantSquare = -1;
     halfMoveClock = 0;
     fullMoveNumber = 1;

     // Parse FEN string
     std::istringstream ss(fen);
     std::string boardPos, activeColor, castling, enPassant, halfMove, fullMove;
     ss >> boardPos >> activeColor >> castling >> enPassant >> halfMove >> fullMove;

     // Set up the board position
     int rank = 7;  // Starting from the 8th rank (index 7)
     int file = 0;  // Starting from the a-file (index 0)

     for (char c : boardPos) {
         if (c == '/') {
             // Move to the next rank
             rank--;
             file = 0;
         }
         else if (c >= '1' && c <= '8') {
             // Skip empty squares
             file += (c - '0');
         }
         else {
             // Place a piece
             int square = rank * 8 + file;
             uint64_t bitboard = 1ULL << square;

             switch (c) {
                 case 'P': whitePawns |= bitboard; break;
                 case 'N': whiteKnights |= bitboard; break;
                 case 'B': whiteBishops |= bitboard; break;
                 case 'R': whiteRooks |= bitboard; break;
                 case 'Q': whiteQueens |= bitboard; break;
                 case 'K': whiteKing |= bitboard; break;
                 case 'p': blackPawns |= bitboard; break;
                 case 'n': blackKnights |= bitboard; break;
                 case 'b': blackBishops |= bitboard; break;
                 case 'r': blackRooks |= bitboard; break;
                 case 'q': blackQueens |= bitboard; break;
                 case 'k': blackKing |= bitboard; break;
             }
             file++;
         }
     }

     // Set active color
     whiteToMove = (activeColor == "w");

     // Set castling rights
     castlingRights = 0;
     if (castling.find('K') != std::string::npos) castlingRights |= 1;     // White kingside
     if (castling.find('Q') != std::string::npos) castlingRights |= 2;     // White queenside
     if (castling.find('k') != std::string::npos) castlingRights |= 4;     // Black kingside
     if (castling.find('q') != std::string::npos) castlingRights |= 8;     // Black queenside

     // Set en passant square
     if (enPassant != "-") {
         file = enPassant[0] - 'a';
         rank = enPassant[1] - '1';
         enPassantSquare = rank * 8 + file;
     }
     else {
         enPassantSquare = -1;
     }

     // Set half-move clock and full move number
     halfMoveClock = std::stoi(halfMove);
     fullMoveNumber = std::stoi(fullMove);
 }

bool ChessEngine::makeMove(const Move& move) {
    // Store piece on target square (for unmake)
    uint64_t toSquareBB = 1ULL << move.to;
    uint64_t fromSquareBB = 1ULL << move.from;

    // Create state for history
    MoveState prevState;
    prevState.enPassantSquare = enPassantSquare;
    prevState.halfMoveClock = halfMoveClock;
    prevState.castlingRights = castlingRights;
    prevState.capturedPieceType = -1;

    // Determine which piece is moving
    uint64_t* movingPiece = nullptr;
    bool isWhite = whiteToMove;

    switch (move.pieceType) {
        case 0: movingPiece = isWhite ? &whitePawns : &blackPawns; break;
        case 1: movingPiece = isWhite ? &whiteKnights : &blackKnights; break;
        case 2: movingPiece = isWhite ? &whiteBishops : &blackBishops; break;
        case 3: movingPiece = isWhite ? &whiteRooks : &blackRooks; break;
        case 4: movingPiece = isWhite ? &whiteQueens : &blackQueens; break;
        case 5: movingPiece = isWhite ? &whiteKing : &blackKing; break;
    }

    if (!movingPiece) return false;

    // Remove the piece from the source square
    *movingPiece &= ~fromSquareBB;

    // Handle captures
    if (move.capture) {
        // Find which piece is being captured
        if (isWhite) {
            if (blackPawns & toSquareBB) {
                prevState.capturedPieceType = 0;
                blackPawns &= ~toSquareBB;
            }
            else if (blackKnights & toSquareBB) {
                prevState.capturedPieceType = 1;
                blackKnights &= ~toSquareBB;
            }
            else if (blackBishops & toSquareBB) {
                prevState.capturedPieceType = 2;
                blackBishops &= ~toSquareBB;
            }
            else if (blackRooks & toSquareBB) {
                prevState.capturedPieceType = 3;
                blackRooks &= ~toSquareBB;
            }
            else if (blackQueens & toSquareBB) {
                prevState.capturedPieceType = 4;
                blackQueens &= ~toSquareBB;
            }
        } else {
            if (whitePawns & toSquareBB) {
                prevState.capturedPieceType = 0;
                whitePawns &= ~toSquareBB;
            }
            else if (whiteKnights & toSquareBB) {
                prevState.capturedPieceType = 1;
                whiteKnights &= ~toSquareBB;
            }
            else if (whiteBishops & toSquareBB) {
                prevState.capturedPieceType = 2;
                whiteBishops &= ~toSquareBB;
            }
            else if (whiteRooks & toSquareBB) {
                prevState.capturedPieceType = 3;
                whiteRooks &= ~toSquareBB;
            }
            else if (whiteQueens & toSquareBB) {
                prevState.capturedPieceType = 4;
                whiteQueens &= ~toSquareBB;
            }
        }
    }

    // Handle special pawn moves
    if (move.pieceType == 0) {
        // En passant capture
        if (move.enPassant) {
            int capturedPawnSquare = move.to + (isWhite ? -8 : 8);
            uint64_t capturedPawnBB = 1ULL << capturedPawnSquare;
            if (isWhite) {
                blackPawns &= ~capturedPawnBB;
            } else {
                whitePawns &= ~capturedPawnBB;
            }
            prevState.capturedPieceType = 0;
        }

        // Set en passant square for double pawn push
        if (move.doublePawnPush) {
            enPassantSquare = move.from + (isWhite ? 8 : -8);
        } else {
            enPassantSquare = -1;
        }

        // Promotions
        if (move.promotion > 0) {
            switch (move.promotion) {
                case 1: // Knight
                    if (isWhite) whiteKnights |= toSquareBB;
                    else blackKnights |= toSquareBB;
                    break;
                case 2: // Bishop
                    if (isWhite) whiteBishops |= toSquareBB;
                    else blackBishops |= toSquareBB;
                    break;
                case 3: // Rook
                    if (isWhite) whiteRooks |= toSquareBB;
                    else blackRooks |= toSquareBB;
                    break;
                case 4: // Queen
                    if (isWhite) whiteQueens |= toSquareBB;
                    else blackQueens |= toSquareBB;
                    break;
            }
        } else {
            *movingPiece |= toSquareBB;
        }
    } else {
        // Clear en passant square for non-pawn moves
        enPassantSquare = -1;

        // Place the piece on its destination (for non-promotion moves)
        *movingPiece |= toSquareBB;
    }

    // Handle castling
    // Handle castling
    if (move.castling && move.pieceType == 5) {
        if (isWhite) {
            if (move.to == 6) { // Kingside
                whiteRooks &= ~(1ULL << 7);  // Remove rook from h1
                whiteRooks |= (1ULL << 5);   // Place rook on f1
            } else if (move.to == 2) { // Queenside
                whiteRooks &= ~(1ULL << 0);  // Remove rook from a1
                whiteRooks |= (1ULL << 3);   // Place rook on d1
            }
        } else {
            if (move.to == 62) { // Black Kingside
                blackRooks &= ~(1ULL << 63); // Remove rook from h8
                blackRooks |= (1ULL << 61);  // Place rook on f8
            } else if (move.to == 58) { // Black Queenside
                blackRooks &= ~(1ULL << 56); // Remove rook from a8
                blackRooks |= (1ULL << 59);  // Place rook on d8
            }
        }
    }

    // Update castling rights
    if (move.pieceType == 5) { // King move
        if (isWhite) {
            castlingRights &= ~3; // Clear white castling rights
        } else {
            castlingRights &= ~12; // Clear black castling rights
        }
    } else if (move.pieceType == 3) { // Rook move
        if (isWhite) {
            if (move.from == 0) castlingRights &= ~2; // a1 rook moved
            if (move.from == 7) castlingRights &= ~1; // h1 rook moved
        } else {
            if (move.from == 56) castlingRights &= ~8; // a8 rook moved
            if (move.from == 63) castlingRights &= ~4; // h8 rook moved
        }
    }

    // If a rook is captured, update castling rights
    if (move.capture) {
        if (move.to == 0) castlingRights &= ~2; // a1 rook captured
        if (move.to == 7) castlingRights &= ~1; // h1 rook captured
        if (move.to == 56) castlingRights &= ~8; // a8 rook captured
        if (move.to == 63) castlingRights &= ~4; // h8 rook captured
    }

    // Update the player to move
    whiteToMove = !whiteToMove;

    // Update move counters
    if (isWhite) {
        fullMoveNumber++;
    }

    if (move.pieceType == 0 || move.capture) {
        halfMoveClock = 0;
    } else {
        halfMoveClock++;
    }

    // Save state history
    stateHistory.push_back(prevState);

    return true;
}

void ChessEngine::unmakeMove(const Move& move) {
    // First, we'll revert the player to move
    whiteToMove = !whiteToMove;
    bool isWhite = whiteToMove;

    // Create bitboards for from and to squares
    uint64_t fromSquareBB = 1ULL << move.from;
    uint64_t toSquareBB = 1ULL << move.to;

    // Determine which piece was moved
    uint64_t* movingPiece = nullptr;
    switch (move.pieceType) {
        case 0: movingPiece = isWhite ? &whitePawns : &blackPawns; break;
        case 1: movingPiece = isWhite ? &whiteKnights : &blackKnights; break;
        case 2: movingPiece = isWhite ? &whiteBishops : &blackBishops; break;
        case 3: movingPiece = isWhite ? &whiteRooks : &blackRooks; break;
        case 4: movingPiece = isWhite ? &whiteQueens : &blackQueens; break;
        case 5: movingPiece = isWhite ? &whiteKing : &blackKing; break;
    }

    // Get previous state from history
    MoveState prevState = stateHistory.back();
    stateHistory.pop_back();

    // Handle promotion
    if (move.promotion > 0) {
        // Remove promoted piece from destination square
        switch (move.promotion) {
            case 1: // Knight
                if (isWhite) whiteKnights &= ~toSquareBB;
                else blackKnights &= ~toSquareBB;
                break;
            case 2: // Bishop
                if (isWhite) whiteBishops &= ~toSquareBB;
                else blackBishops &= ~toSquareBB;
                break;
            case 3: // Rook
                if (isWhite) whiteRooks &= ~toSquareBB;
                else blackRooks &= ~toSquareBB;
                break;
            case 4: // Queen
                if (isWhite) whiteQueens &= ~toSquareBB;
                else blackQueens &= ~toSquareBB;
                break;
        }
        // Put the pawn back on the origin square
        *movingPiece |= fromSquareBB;
    } else {
        // Regular piece move: remove from destination, put back at origin
        if (!move.castling) {
            *movingPiece &= ~toSquareBB;
        }
        *movingPiece |= fromSquareBB;
    }

    // Handle captures - restore the captured piece
    if (move.capture) {
        // For en passant, the captured pawn isn't on the destination square
        if (move.enPassant) {
            int capturedPawnSquare = move.to + (isWhite ? -8 : 8);
            uint64_t capturedPawnBB = 1ULL << capturedPawnSquare;
            if (isWhite) {
                blackPawns |= capturedPawnBB;
            } else {
                whitePawns |= capturedPawnBB;
            }
        } else {
            if (isWhite) { // White captured a black piece
                switch (prevState.capturedPieceType) {
                    case 0: blackPawns |= toSquareBB; break;
                    case 1: blackKnights |= toSquareBB; break;
                    case 2: blackBishops |= toSquareBB; break;
                    case 3: blackRooks |= toSquareBB; break;
                    case 4: blackQueens |= toSquareBB; break;
                }
            } else { // Black captured a white piece
                switch (prevState.capturedPieceType) {
                    case 0: whitePawns |= toSquareBB; break;
                    case 1: whiteKnights |= toSquareBB; break;
                    case 2: whiteBishops |= toSquareBB; break;
                    case 3: whiteRooks |= toSquareBB; break;
                    case 4: whiteQueens |= toSquareBB; break;
                }
            }
        }
    }

    // Handle castling
    if (move.castling && move.pieceType == 5) {
        if (isWhite) {
            if (move.to == 6) { // Kingside
                whiteRooks &= ~(1ULL << 5);  // Remove rook from f1
                whiteRooks |= (1ULL << 7);   // Place rook back on h1
            } else if (move.to == 2) { // Queenside
                whiteRooks &= ~(1ULL << 3);  // Remove rook from d1
                whiteRooks |= (1ULL << 0);   // Place rook back on a1
            }
        } else {
            if (move.to == 62) { // Kingside
                blackRooks &= ~(1ULL << 61);  // Remove rook from f8
                blackRooks |= (1ULL << 63);   // Place rook back on h8
            } else if (move.to == 58) { // Queenside
                blackRooks &= ~(1ULL << 59);  // Remove rook from d8
                blackRooks |= (1ULL << 56);   // Place rook back on a8
            }
        }
    }

    // Restore previous state
    enPassantSquare = prevState.enPassantSquare;
    halfMoveClock = prevState.halfMoveClock;
    castlingRights = prevState.castlingRights;

    // Restore move counters
    if (isWhite) {
        fullMoveNumber--;
    }
}

bool ChessEngine::isInCheck(bool isWhite) {
    // Find the king's position
    uint64_t kingBB = isWhite ? whiteKing : blackKing;
    int kingSquare = bitScanForward(kingBB);

    if (kingSquare == -1) return false; // No king (shouldn't happen)

    // Check if the king's square is attacked by the opponent
    return isSquareAttacked(kingSquare, !isWhite);
}

// Implementation for Move::toString
std::string ChessEngine::Move::toString() const {
    // Convert move to algebraic notation
    std::string result;
    char fileFrom = 'a' + (from % 8);
    char rankFrom = '1' + (from / 8);
    char fileTo = 'a' + (to % 8);
    char rankTo = '1' + (to / 8);

    result += fileFrom;
    result += rankFrom;
    result += fileTo;
    result += rankTo;

    // Add promotion piece if applicable
    if (promotion > 0) {
        char promotionPiece = ' ';
        switch (promotion) {
            case 1: promotionPiece = 'n'; break; // knight
            case 2: promotionPiece = 'b'; break; // bishop
            case 3: promotionPiece = 'r'; break; // rook
            case 4: promotionPiece = 'q'; break; // queen
        }
        result += promotionPiece;
    }

    return result;
}

void ChessEngine::printBoard() const {
    std::cout << "\n  +---+---+---+---+---+---+---+---+\n";

    for (int rank = 7; rank >= 0; rank--) {
        std::cout << (rank + 1) << " |";

        for (int file = 0; file < 8; file++) {
            int square = rank * 8 + file;
            uint64_t squareBit = 1ULL << square;
            char piece = ' ';

            // Check which piece is on this square
            if (whitePawns & squareBit) piece = 'P';
            else if (whiteKnights & squareBit) piece = 'N';
            else if (whiteBishops & squareBit) piece = 'B';
            else if (whiteRooks & squareBit) piece = 'R';
            else if (whiteQueens & squareBit) piece = 'Q';
            else if (whiteKing & squareBit) piece = 'K';
            else if (blackPawns & squareBit) piece = 'p';
            else if (blackKnights & squareBit) piece = 'n';
            else if (blackBishops & squareBit) piece = 'b';
            else if (blackRooks & squareBit) piece = 'r';
            else if (blackQueens & squareBit) piece = 'q';
            else if (blackKing & squareBit) piece = 'k';

            std::cout << " " << piece << " |";
        }

        std::cout << "\n  +---+---+---+---+---+---+---+---+\n";
    }

    std::cout << "    a   b   c   d   e   f   g   h\n";
}

// Add a main function to make the program executable
int main(int argc, char* argv[]) {
    ChessEngine engine;
    std::string userInput;

    // Check if printBoard parameter was provided
    bool shouldPrintBoard = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "printBoard") {
            shouldPrintBoard = true;
            break;
        }
    }
    // Default FEN for the starting position
    const std::string defaultFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    // Print initial board state
    if (shouldPrintBoard){
        engine.printBoard();
    }

    bool whiteToPlay = true;

    while (true) {
        // Display prompt
        std::cout << "\n" << (whiteToPlay ? "White" : "Black") << " to move > ";

        // Get user input
        std::getline(std::cin, userInput);

        // Handle special commands
        if (userInput == "quit" || userInput == "exit") {
            break;
        }
        else if (userInput == "help") {
            std::cout << "Commands:\n";
            std::cout << "  - Enter moves in format 'e2e4'\n";
            std::cout << "  - 'moves' - Show legal moves\n";
            std::cout << "  - 'board' - Show the board\n";
            std::cout << "  - 'fen [FEN string]' - Set position from FEN\n";
            std::cout << "  - 'reset' - Reset to starting position\n";
            std::cout << "  - 'quit'  - Exit the program\n";
            continue;
        }
        else if (userInput == "board") {
            engine.printBoard();
            continue;
        }
        else if (userInput == "moves") {
            auto moves = engine.getLegalMoves(whiteToPlay);
            std::cout << "Legal moves: " << moves.size() << "\n";
            for (const auto& move : moves) {
                std::cout << move.toString() << " ";
            }
            std::cout << "\n";
            continue;
        }
        else if (userInput == "reset") {
            engine.setPosition(defaultFEN);
            whiteToPlay = true;
            engine.printBoard();
            continue;
        }
        else if (userInput.substr(0, 4) == "fen ") {
            std::string fenString = userInput.substr(4);  // Extract FEN string
            if (fenString.empty()) {
                std::cout << "Using default starting position\n";
                fenString = defaultFEN;
            }

            try {
                engine.setPosition(fenString);
                // Update whose turn it is based on the FEN
                whiteToPlay = (fenString.find(" w ") != std::string::npos);
                std::cout << "Position set from FEN\n";
                engine.printBoard();
            }
            catch (const std::exception& e) {
                std::cout << "Error setting position: " << e.what() << "\n";
            }
            continue;
        }

        // Try to parse as a move (e.g. "e2e4")
        if (userInput.length() >= 4) {
            // Convert algebraic notation to move
            int fileFrom = userInput[0] - 'a';
            int rankFrom = userInput[1] - '1';
            int fileTo = userInput[2] - 'a';
            int rankTo = userInput[3] - '1';

            if (fileFrom >= 0 && fileFrom < 8 &&
                rankFrom >= 0 && rankFrom < 8 &&
                fileTo >= 0 && fileTo < 8 &&
                rankTo >= 0 && rankTo < 8) {

                int fromSquare = rankFrom * 8 + fileFrom;
                int toSquare = rankTo * 8 + fileTo;

                // Find the move in the legal moves list
                auto legalMoves = engine.getLegalMoves(whiteToPlay);
                bool moveFound = false;

                for (const auto& move : legalMoves) {
                    if (move.from == fromSquare && move.to == toSquare) {
                        // Handle promotion
                        if (move.promotion > 0 && userInput.length() >= 5) {
                            char promo = userInput[4];
                            int promoType = 0;
                            switch(promo) {
                                case 'n': promoType = 1; break;
                                case 'b': promoType = 2; break;
                                case 'r': promoType = 3; break;
                                case 'q': promoType = 4; break;
                            }
                            if (move.promotion != promoType) continue;
                        }

                        if (engine.makeMove(move)) {
                            std::cout << "Move: " << move.toString() << "\n";
                            if (shouldPrintBoard){
                                engine.printBoard();
                            }
                            whiteToPlay = !whiteToPlay;
                            moveFound = true;
                            break;
                        }
                    }
                }

                if (!moveFound) {
                    std::cout << "Illegal move.\n";
                }
            }
        } else {
            std::cout << "Invalid input. Type 'help' for commands.\n";
        }
    }

    return 0;
}