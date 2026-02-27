import { Request, Response, NextFunction } from 'express';
import { validateMarketDataRequest } from '../../middleware/validation/marketDataValidation';

describe('Market Data Validation Middleware', () => {
  let mockRequest: Partial<Request>;
  let mockResponse: Partial<Response>;
  let nextFunction: NextFunction;

  beforeEach(() => {
    mockRequest = {
      params: {},
      query: {},
    };
    mockResponse = {
      status: jest.fn().mockReturnThis(),
      json: jest.fn(),
    };
    nextFunction = jest.fn();
  });

  // Issue #404 Stream C: Updated tests for multi-symbol support
  describe('Symbol Parameter Validation', () => {
    it('should pass validation for valid symbol EURUSD', () => {
      mockRequest.params = { symbol: 'EURUSD', timeframe: 'H1' };
      mockRequest.query = {};

      validateMarketDataRequest(mockRequest as Request, mockResponse as Response, nextFunction);

      expect(nextFunction).toHaveBeenCalled();
      expect(mockResponse.status).not.toHaveBeenCalled();
    });

    // Issue #404 Stream C: Test all 5 supported symbols
    it('should pass validation for all supported symbols', () => {
      const supportedSymbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'EURJPY', 'USDCAD', 'USDCHF', 'EURCAD', 'EURGBP'];

      for (const symbol of supportedSymbols) {
        jest.clearAllMocks();
        mockRequest.params = { symbol, timeframe: 'H1' };
        mockRequest.query = {};

        validateMarketDataRequest(mockRequest as Request, mockResponse as Response, nextFunction);

        expect(nextFunction).toHaveBeenCalled();
        expect(mockResponse.status).not.toHaveBeenCalled();
      }
    });

    it('should return 400 for invalid symbol', () => {
      mockRequest.params = { symbol: 'INVALID_SYMBOL', timeframe: 'H1' };
      mockRequest.query = {};

      validateMarketDataRequest(mockRequest as Request, mockResponse as Response, nextFunction);

      expect(mockResponse.status).toHaveBeenCalledWith(400);
      expect(mockResponse.json).toHaveBeenCalledWith({
        error: {
          code: 'VALIDATION_ERROR',
          message: expect.stringContaining('symbol'),
          details: expect.objectContaining({
            symbol: 'INVALID_SYMBOL',
            valid: ['EURUSD', 'GBPUSD', 'USDJPY', 'EURJPY', 'USDCAD', 'USDCHF', 'EURCAD', 'EURGBP'],
          }),
        },
      });
      expect(nextFunction).not.toHaveBeenCalled();
    });

    it('should return 400 for missing symbol', () => {
      mockRequest.params = { timeframe: 'H1' };
      mockRequest.query = {};

      validateMarketDataRequest(mockRequest as Request, mockResponse as Response, nextFunction);

      expect(mockResponse.status).toHaveBeenCalledWith(400);
      expect(mockResponse.json).toHaveBeenCalledWith({
        error: {
          code: 'VALIDATION_ERROR',
          message: expect.stringContaining('symbol'),
          details: expect.any(Object),
        },
      });
      expect(nextFunction).not.toHaveBeenCalled();
    });
  });

  // Issue #404 Stream C: Updated tests for all 12 supported timeframes
  describe('Timeframe Parameter Validation', () => {
    it('should pass validation for valid timeframe H1', () => {
      mockRequest.params = { symbol: 'EURUSD', timeframe: 'H1' };
      mockRequest.query = {};

      validateMarketDataRequest(mockRequest as Request, mockResponse as Response, nextFunction);

      expect(nextFunction).toHaveBeenCalled();
      expect(mockResponse.status).not.toHaveBeenCalled();
    });

    it('should pass validation for valid timeframe H4', () => {
      mockRequest.params = { symbol: 'EURUSD', timeframe: 'H4' };
      mockRequest.query = {};

      validateMarketDataRequest(mockRequest as Request, mockResponse as Response, nextFunction);

      expect(nextFunction).toHaveBeenCalled();
      expect(mockResponse.status).not.toHaveBeenCalled();
    });

    it('should pass validation for valid timeframe D1', () => {
      mockRequest.params = { symbol: 'EURUSD', timeframe: 'D1' };
      mockRequest.query = {};

      validateMarketDataRequest(mockRequest as Request, mockResponse as Response, nextFunction);

      expect(nextFunction).toHaveBeenCalled();
      expect(mockResponse.status).not.toHaveBeenCalled();
    });

    // Issue #404 Stream C: Test all 12 supported timeframes
    it('should pass validation for all supported timeframes', () => {
      const supportedTimeframes = [
        'M1',
        'M5',
        'M15',
        'M30',
        'H1',
        'H2',
        'H3',
        'H4',
        'H6',
        'H8',
        'H12',
        'D1',
      ];

      for (const timeframe of supportedTimeframes) {
        jest.clearAllMocks();
        mockRequest.params = { symbol: 'EURUSD', timeframe };
        mockRequest.query = {};

        validateMarketDataRequest(mockRequest as Request, mockResponse as Response, nextFunction);

        expect(nextFunction).toHaveBeenCalled();
        expect(mockResponse.status).not.toHaveBeenCalled();
      }
    });

    it('should return 400 for invalid timeframe', () => {
      mockRequest.params = { symbol: 'EURUSD', timeframe: 'INVALID' };
      mockRequest.query = {};

      validateMarketDataRequest(mockRequest as Request, mockResponse as Response, nextFunction);

      expect(mockResponse.status).toHaveBeenCalledWith(400);
      expect(mockResponse.json).toHaveBeenCalledWith({
        error: {
          code: 'VALIDATION_ERROR',
          message: expect.stringContaining('timeframe'),
          details: expect.objectContaining({
            timeframe: 'INVALID',
            valid: ['M1', 'M5', 'M15', 'M30', 'H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'H12', 'D1'],
          }),
        },
      });
      expect(nextFunction).not.toHaveBeenCalled();
    });

    it('should return 400 for missing timeframe', () => {
      mockRequest.params = { symbol: 'EURUSD' };
      mockRequest.query = {};

      validateMarketDataRequest(mockRequest as Request, mockResponse as Response, nextFunction);

      expect(mockResponse.status).toHaveBeenCalledWith(400);
      expect(mockResponse.json).toHaveBeenCalledWith({
        error: {
          code: 'VALIDATION_ERROR',
          message: expect.stringContaining('timeframe'),
          details: expect.any(Object),
        },
      });
      expect(nextFunction).not.toHaveBeenCalled();
    });
  });

  describe('Query Parameter Validation - Start Date', () => {
    it('should pass validation for valid ISO 8601 start date', () => {
      mockRequest.params = { symbol: 'EURUSD', timeframe: 'H1' };
      mockRequest.query = { start: '2024-01-01T00:00:00Z' };

      validateMarketDataRequest(mockRequest as Request, mockResponse as Response, nextFunction);

      expect(nextFunction).toHaveBeenCalled();
      expect(mockResponse.status).not.toHaveBeenCalled();
    });

    it('should return 400 for invalid start date format', () => {
      mockRequest.params = { symbol: 'EURUSD', timeframe: 'H1' };
      mockRequest.query = { start: 'invalid-date' };

      validateMarketDataRequest(mockRequest as Request, mockResponse as Response, nextFunction);

      expect(mockResponse.status).toHaveBeenCalledWith(400);
      expect(mockResponse.json).toHaveBeenCalledWith({
        error: {
          code: 'VALIDATION_ERROR',
          message: expect.stringContaining('ISO 8601'),
          details: expect.any(Object),
        },
      });
      expect(nextFunction).not.toHaveBeenCalled();
    });

    it('should return 400 for start date in the future', () => {
      mockRequest.params = { symbol: 'EURUSD', timeframe: 'H1' };
      const futureDate = new Date();
      futureDate.setDate(futureDate.getDate() + 1);
      mockRequest.query = { start: futureDate.toISOString() };

      validateMarketDataRequest(mockRequest as Request, mockResponse as Response, nextFunction);

      expect(mockResponse.status).toHaveBeenCalledWith(400);
      expect(mockResponse.json).toHaveBeenCalledWith({
        error: {
          code: 'VALIDATION_ERROR',
          message: expect.stringContaining('future'),
          details: expect.any(Object),
        },
      });
      expect(nextFunction).not.toHaveBeenCalled();
    });
  });

  describe('Query Parameter Validation - End Date', () => {
    it('should pass validation for valid ISO 8601 end date', () => {
      mockRequest.params = { symbol: 'EURUSD', timeframe: 'H1' };
      mockRequest.query = { end: '2024-01-31T23:59:59Z' };

      validateMarketDataRequest(mockRequest as Request, mockResponse as Response, nextFunction);

      expect(nextFunction).toHaveBeenCalled();
      expect(mockResponse.status).not.toHaveBeenCalled();
    });

    it('should return 400 for invalid end date format', () => {
      mockRequest.params = { symbol: 'EURUSD', timeframe: 'H1' };
      mockRequest.query = { end: 'not-a-date' };

      validateMarketDataRequest(mockRequest as Request, mockResponse as Response, nextFunction);

      expect(mockResponse.status).toHaveBeenCalledWith(400);
      expect(mockResponse.json).toHaveBeenCalledWith({
        error: {
          code: 'VALIDATION_ERROR',
          message: expect.stringContaining('ISO 8601'),
          details: expect.any(Object),
        },
      });
      expect(nextFunction).not.toHaveBeenCalled();
    });

    it('should return 400 for end date in the future', () => {
      mockRequest.params = { symbol: 'EURUSD', timeframe: 'H1' };
      const futureDate = new Date();
      futureDate.setDate(futureDate.getDate() + 1);
      mockRequest.query = { end: futureDate.toISOString() };

      validateMarketDataRequest(mockRequest as Request, mockResponse as Response, nextFunction);

      expect(mockResponse.status).toHaveBeenCalledWith(400);
      expect(mockResponse.json).toHaveBeenCalledWith({
        error: {
          code: 'VALIDATION_ERROR',
          message: expect.stringContaining('future'),
          details: expect.any(Object),
        },
      });
      expect(nextFunction).not.toHaveBeenCalled();
    });

    it('should return 400 when end date is before start date', () => {
      mockRequest.params = { symbol: 'EURUSD', timeframe: 'H1' };
      mockRequest.query = {
        start: '2024-01-31T00:00:00Z',
        end: '2024-01-01T00:00:00Z',
      };

      validateMarketDataRequest(mockRequest as Request, mockResponse as Response, nextFunction);

      expect(mockResponse.status).toHaveBeenCalledWith(400);
      expect(mockResponse.json).toHaveBeenCalledWith({
        error: {
          code: 'VALIDATION_ERROR',
          message: expect.stringContaining('end date must be greater than or equal to start date'),
          details: expect.any(Object),
        },
      });
      expect(nextFunction).not.toHaveBeenCalled();
    });

    it('should pass validation when end date equals start date', () => {
      mockRequest.params = { symbol: 'EURUSD', timeframe: 'H1' };
      mockRequest.query = {
        start: '2024-01-01T00:00:00Z',
        end: '2024-01-01T00:00:00Z',
      };

      validateMarketDataRequest(mockRequest as Request, mockResponse as Response, nextFunction);

      expect(nextFunction).toHaveBeenCalled();
      expect(mockResponse.status).not.toHaveBeenCalled();
    });
  });

  describe('Query Parameter Validation - Limit', () => {
    it('should pass validation for valid limit within range', () => {
      mockRequest.params = { symbol: 'EURUSD', timeframe: 'H1' };
      mockRequest.query = { limit: '1000' };

      validateMarketDataRequest(mockRequest as Request, mockResponse as Response, nextFunction);

      expect(nextFunction).toHaveBeenCalled();
      expect(mockResponse.status).not.toHaveBeenCalled();
    });

    it('should pass validation for limit = 1 (minimum)', () => {
      mockRequest.params = { symbol: 'EURUSD', timeframe: 'H1' };
      mockRequest.query = { limit: '1' };

      validateMarketDataRequest(mockRequest as Request, mockResponse as Response, nextFunction);

      expect(nextFunction).toHaveBeenCalled();
      expect(mockResponse.status).not.toHaveBeenCalled();
    });

    it('should pass validation for limit = 5000 (maximum)', () => {
      mockRequest.params = { symbol: 'EURUSD', timeframe: 'H1' };
      mockRequest.query = { limit: '5000' };

      validateMarketDataRequest(mockRequest as Request, mockResponse as Response, nextFunction);

      expect(nextFunction).toHaveBeenCalled();
      expect(mockResponse.status).not.toHaveBeenCalled();
    });

    it('should return 400 for limit below minimum (0)', () => {
      mockRequest.params = { symbol: 'EURUSD', timeframe: 'H1' };
      mockRequest.query = { limit: '0' };

      validateMarketDataRequest(mockRequest as Request, mockResponse as Response, nextFunction);

      expect(mockResponse.status).toHaveBeenCalledWith(400);
      expect(mockResponse.json).toHaveBeenCalledWith({
        error: {
          code: 'VALIDATION_ERROR',
          message: expect.stringContaining('between'),
          details: expect.objectContaining({
            min: 1,
            max: 5000,
          }),
        },
      });
      expect(nextFunction).not.toHaveBeenCalled();
    });

    it('should return 400 for limit above maximum (5001)', () => {
      mockRequest.params = { symbol: 'EURUSD', timeframe: 'H1' };
      mockRequest.query = { limit: '5001' };

      validateMarketDataRequest(mockRequest as Request, mockResponse as Response, nextFunction);

      expect(mockResponse.status).toHaveBeenCalledWith(400);
      expect(mockResponse.json).toHaveBeenCalledWith({
        error: {
          code: 'VALIDATION_ERROR',
          message: expect.stringContaining('between'),
          details: expect.objectContaining({
            min: 1,
            max: 5000,
          }),
        },
      });
      expect(nextFunction).not.toHaveBeenCalled();
    });

    it('should return 400 for non-numeric limit', () => {
      mockRequest.params = { symbol: 'EURUSD', timeframe: 'H1' };
      mockRequest.query = { limit: 'abc' };

      validateMarketDataRequest(mockRequest as Request, mockResponse as Response, nextFunction);

      expect(mockResponse.status).toHaveBeenCalledWith(400);
      expect(mockResponse.json).toHaveBeenCalledWith({
        error: {
          code: 'VALIDATION_ERROR',
          message: expect.stringContaining('between'),
          details: expect.any(Object),
        },
      });
      expect(nextFunction).not.toHaveBeenCalled();
    });

    it('should pass validation when limit is not provided (use default)', () => {
      mockRequest.params = { symbol: 'EURUSD', timeframe: 'H1' };
      mockRequest.query = {};

      validateMarketDataRequest(mockRequest as Request, mockResponse as Response, nextFunction);

      expect(nextFunction).toHaveBeenCalled();
      expect(mockResponse.status).not.toHaveBeenCalled();
    });
  });

  describe('Multiple Validation Errors', () => {
    it('should return all validation errors when multiple parameters are invalid', () => {
      mockRequest.params = { symbol: 'INVALID', timeframe: 'INVALID' };
      mockRequest.query = { limit: '-1' };

      validateMarketDataRequest(mockRequest as Request, mockResponse as Response, nextFunction);

      expect(mockResponse.status).toHaveBeenCalledWith(400);
      expect(mockResponse.json).toHaveBeenCalledWith({
        error: {
          code: 'VALIDATION_ERROR',
          message: expect.any(String),
          details: expect.any(Object),
        },
      });
      expect(nextFunction).not.toHaveBeenCalled();
    });
  });
});
