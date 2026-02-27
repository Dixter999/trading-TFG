import { Request, Response, NextFunction } from 'express';
import { formatResponse } from '../../middleware/responseFormatter';

// Extend Request type to include id
interface RequestWithId extends Request {
  id?: string;
}

describe('Response Formatter Middleware', () => {
  let mockRequest: Partial<RequestWithId>;
  let mockResponse: Partial<Response>;
  let nextFunction: NextFunction;

  beforeEach(() => {
    mockRequest = {
      id: 'req_test123',
    };
    mockResponse = {
      json: jest.fn().mockReturnThis(),
      status: jest.fn().mockReturnThis(),
      locals: {},
    };
    nextFunction = jest.fn();
  });

  describe('Data transformation', () => {
    it('should format database results to match PRD specification', () => {
      const dbResults = [
        {
          rate_time: '2024-01-15T12:00:00.000Z',
          open: '1.0850',
          high: '1.0875',
          low: '1.0840',
          close: '1.0865',
          volume: 1500,
          sma_20: '1.0855',
          sma_50: '1.0860',
          sma_200: '1.0870',
          ema_12: '1.0858',
          ema_26: '1.0862',
          ema_50: '1.0865',
          rsi_14: '55.3',
          atr_14: '0.0025',
          bb_upper_20: '1.0890',
          bb_middle_20: '1.0855',
          bb_lower_20: '1.0820',
          macd_line: '0.0003',
          macd_signal: '0.0002',
          macd_histogram: '0.0001',
          stoch_k: '60.5',
          stoch_d: '58.2',
        },
      ];

      mockResponse.locals = { data: dbResults };

      formatResponse(mockRequest as RequestWithId, mockResponse as Response, nextFunction);

      expect(mockResponse.json).toHaveBeenCalledWith(
        expect.objectContaining({
          data: expect.arrayContaining([
            expect.objectContaining({
              rateTime: '2024-01-15T12:00:00.000Z',
              open: 1.085,
              high: 1.0875,
              low: 1.084,
              close: 1.0865,
              volume: 1500,
              indicators: expect.objectContaining({
                sma_20: 1.0855,
                sma_50: 1.086,
                sma_200: 1.087,
                ema_12: 1.0858,
                ema_26: 1.0862,
                ema_50: 1.0865,
                rsi_14: 55.3,
                atr_14: 0.0025,
                bb_upper_20: 1.089,
                bb_middle_20: 1.0855,
                bb_lower_20: 1.082,
                macd_line: 0.0003,
                macd_signal: 0.0002,
                macd_histogram: 0.0001,
                stoch_k: 60.5,
                stoch_d: 58.2,
              }),
            }),
          ]),
        })
      );
    });

    it('should convert string numeric values to numbers', () => {
      const dbResults = [
        {
          rate_time: '2024-01-15T12:00:00.000Z',
          open: '1.0850',
          high: '1.0875',
          low: '1.0840',
          close: '1.0865',
          volume: 1500,
          sma_20: '1.0855',
        },
      ];

      mockResponse.locals = { data: dbResults };

      formatResponse(mockRequest as RequestWithId, mockResponse as Response, nextFunction);

      const callArg = (mockResponse.json as jest.Mock).mock.calls[0][0];
      expect(typeof callArg.data[0].open).toBe('number');
      expect(typeof callArg.data[0].high).toBe('number');
      expect(typeof callArg.data[0].low).toBe('number');
      expect(typeof callArg.data[0].close).toBe('number');
      expect(typeof callArg.data[0].indicators.sma_20).toBe('number');
    });

    it('should rename rate_time to rateTime in OHLCV data', () => {
      const dbResults = [
        {
          rate_time: '2024-01-15T12:00:00.000Z',
          open: '1.0850',
          high: '1.0875',
          low: '1.0840',
          close: '1.0865',
          volume: 1500,
        },
      ];

      mockResponse.locals = { data: dbResults };

      formatResponse(mockRequest as RequestWithId, mockResponse as Response, nextFunction);

      const callArg = (mockResponse.json as jest.Mock).mock.calls[0][0];
      expect(callArg.data[0]).toHaveProperty('rateTime');
      expect(callArg.data[0]).not.toHaveProperty('rate_time');
    });

    it('should group indicator fields into indicators object', () => {
      const dbResults = [
        {
          rate_time: '2024-01-15T12:00:00.000Z',
          open: '1.0850',
          high: '1.0875',
          low: '1.0840',
          close: '1.0865',
          volume: 1500,
          sma_20: '1.0855',
          ema_12: '1.0858',
          rsi_14: '55.3',
        },
      ];

      mockResponse.locals = { data: dbResults };

      formatResponse(mockRequest as RequestWithId, mockResponse as Response, nextFunction);

      const callArg = (mockResponse.json as jest.Mock).mock.calls[0][0];
      expect(callArg.data[0]).toHaveProperty('indicators');
      expect(callArg.data[0].indicators).toHaveProperty('sma_20');
      expect(callArg.data[0].indicators).toHaveProperty('ema_12');
      expect(callArg.data[0].indicators).toHaveProperty('rsi_14');
      expect(callArg.data[0]).not.toHaveProperty('sma_20');
      expect(callArg.data[0]).not.toHaveProperty('ema_12');
      expect(callArg.data[0]).not.toHaveProperty('rsi_14');
    });
  });

  describe('Metadata addition', () => {
    it('should add metadata with count field', () => {
      const dbResults = [
        {
          rate_time: '2024-01-15T12:00:00.000Z',
          open: '1.0850',
          high: '1.0875',
          low: '1.0840',
          close: '1.0865',
          volume: 1500,
        },
      ];

      mockResponse.locals = { data: dbResults };

      formatResponse(mockRequest as RequestWithId, mockResponse as Response, nextFunction);

      expect(mockResponse.json).toHaveBeenCalledWith(
        expect.objectContaining({
          metadata: expect.objectContaining({
            count: 1,
          }),
        })
      );
    });

    it('should add metadata with ISO 8601 timestamp', () => {
      const dbResults = [
        {
          rate_time: '2024-01-15T12:00:00.000Z',
          open: '1.0850',
          high: '1.0875',
          low: '1.0840',
          close: '1.0865',
          volume: 1500,
        },
      ];

      mockResponse.locals = { data: dbResults };

      formatResponse(mockRequest as RequestWithId, mockResponse as Response, nextFunction);

      const callArg = (mockResponse.json as jest.Mock).mock.calls[0][0];
      expect(callArg.metadata).toHaveProperty('timestamp');
      expect(typeof callArg.metadata.timestamp).toBe('string');

      // Verify it's a valid ISO 8601 date
      const timestamp = new Date(callArg.metadata.timestamp);
      expect(timestamp.toString()).not.toBe('Invalid Date');
    });

    it('should add metadata with requestId from request', () => {
      const dbResults = [
        {
          rate_time: '2024-01-15T12:00:00.000Z',
          open: '1.0850',
          high: '1.0875',
          low: '1.0840',
          close: '1.0865',
          volume: 1500,
        },
      ];

      mockResponse.locals = { data: dbResults };
      mockRequest.id = 'req_abc123';

      formatResponse(mockRequest as RequestWithId, mockResponse as Response, nextFunction);

      expect(mockResponse.json).toHaveBeenCalledWith(
        expect.objectContaining({
          metadata: expect.objectContaining({
            requestId: 'req_abc123',
          }),
        })
      );
    });
  });

  describe('Edge cases', () => {
    it('should handle empty data array', () => {
      mockResponse.locals = { data: [] };

      formatResponse(mockRequest as RequestWithId, mockResponse as Response, nextFunction);

      expect(mockResponse.json).toHaveBeenCalledWith(
        expect.objectContaining({
          data: [],
          metadata: expect.objectContaining({
            count: 0,
          }),
        })
      );
    });

    it('should call next() if no data in response.locals', () => {
      mockResponse.locals = {};

      formatResponse(mockRequest as RequestWithId, mockResponse as Response, nextFunction);

      expect(nextFunction).toHaveBeenCalled();
      expect(mockResponse.json).not.toHaveBeenCalled();
    });

    it('should handle null indicator values', () => {
      const dbResults = [
        {
          rate_time: '2024-01-15T12:00:00.000Z',
          open: '1.0850',
          high: '1.0875',
          low: '1.0840',
          close: '1.0865',
          volume: 1500,
          sma_20: null,
          ema_12: null,
        },
      ];

      mockResponse.locals = { data: dbResults };

      formatResponse(mockRequest as RequestWithId, mockResponse as Response, nextFunction);

      const callArg = (mockResponse.json as jest.Mock).mock.calls[0][0];
      expect(callArg.data[0].indicators.sma_20).toBeNull();
      expect(callArg.data[0].indicators.ema_12).toBeNull();
    });
  });
});
