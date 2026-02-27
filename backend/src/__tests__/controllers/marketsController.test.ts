/**
 * Markets Controller Tests
 *
 * TDD: RED-GREEN-REFACTOR
 * These tests are written FIRST before implementation
 */

import { Request, Response, NextFunction } from 'express';
import { getMarketData } from '../../controllers/marketsController';
import * as marketDataQueries from '../../database/queries/marketData';
import { ValidationError, QueryError } from '../../database/errors';

// Mock the database layer
jest.mock('../../database/queries/marketData');

describe('Markets Controller', () => {
  let mockRequest: Partial<Request>;
  let mockResponse: Partial<Response>;
  let mockNext: NextFunction;

  beforeEach(() => {
    // Reset mocks before each test
    jest.clearAllMocks();

    // Setup mock response with locals
    mockResponse = {
      locals: {},
    };

    mockNext = jest.fn();

    // Mock helper functions for optional parameters
    // Default to returning reasonable timestamps
    (marketDataQueries.getEarliestTimestamp as jest.Mock).mockResolvedValue(
      new Date('2024-01-01T00:00:00Z')
    );
    (marketDataQueries.getLatestTimestamp as jest.Mock).mockResolvedValue(
      new Date('2024-12-31T23:59:59Z')
    );
  });

  describe('getMarketData - Basic Functionality', () => {
    it('should return market data with indicators for valid request', async () => {
      // Arrange
      mockRequest = {
        params: {
          symbol: 'EURUSD',
          timeframe: 'H1',
        },
        query: {
          start: '2024-01-01T00:00:00Z',
          end: '2024-01-31T23:59:59Z',
          limit: '1000',
        },
      };

      const mockData = [
        {
          rateTime: new Date('2024-01-15T12:00:00Z'),
          open: 1.085,
          high: 1.0875,
          low: 1.084,
          close: 1.0865,
          volume: 1500,
          indicators: {
            sma_20: 1.0855,
            rsi_14: 55.3,
          },
        },
      ];

      (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockResolvedValue(mockData);

      // Act
      await getMarketData(mockRequest as Request, mockResponse as Response, mockNext);

      // Assert
      expect(marketDataQueries.getMarketDataWithIndicators).toHaveBeenCalledWith(
        'EURUSD',
        'H1',
        new Date('2024-01-01T00:00:00Z'),
        new Date('2024-01-31T23:59:59Z'),
        1000
      );
      expect(mockResponse.locals?.['data']).toEqual(mockData);
      expect(mockNext).toHaveBeenCalledWith();
    });

    it('should use default limit when not provided', async () => {
      // Arrange
      mockRequest = {
        params: {
          symbol: 'EURUSD',
          timeframe: 'H1',
        },
        query: {
          start: '2024-01-01T00:00:00Z',
          end: '2024-01-31T23:59:59Z',
        },
      };

      (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockResolvedValue([]);

      // Act
      await getMarketData(mockRequest as Request, mockResponse as Response, mockNext);

      // Assert
      expect(marketDataQueries.getMarketDataWithIndicators).toHaveBeenCalledWith(
        'EURUSD',
        'H1',
        expect.any(Date),
        expect.any(Date),
        1000 // default limit
      );
      expect(mockNext).toHaveBeenCalledWith();
    });
  });

  describe('getMarketData - Query Parameter Parsing', () => {
    it('should parse ISO 8601 date strings correctly', async () => {
      // Arrange
      mockRequest = {
        params: {
          symbol: 'EURUSD',
          timeframe: 'H4',
        },
        query: {
          start: '2024-01-01T00:00:00.000Z',
          end: '2024-12-31T23:59:59.999Z',
          limit: '500',
        },
      };

      (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockResolvedValue([]);

      // Act
      await getMarketData(mockRequest as Request, mockResponse as Response, mockNext);

      // Assert
      expect(marketDataQueries.getMarketDataWithIndicators).toHaveBeenCalledWith(
        'EURUSD',
        'H4',
        new Date('2024-01-01T00:00:00.000Z'),
        new Date('2024-12-31T23:59:59.999Z'),
        500
      );
    });

    it('should parse limit as number', async () => {
      // Arrange
      mockRequest = {
        params: {
          symbol: 'EURUSD',
          timeframe: 'D1',
        },
        query: {
          start: '2024-01-01T00:00:00Z',
          end: '2024-01-31T23:59:59Z',
          limit: '5000',
        },
      };

      (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockResolvedValue([]);

      // Act
      await getMarketData(mockRequest as Request, mockResponse as Response, mockNext);

      // Assert
      const call = (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mock.calls[0];
      expect(call[4]).toBe(5000);
      expect(typeof call[4]).toBe('number');
    });

    it('should handle missing query parameters gracefully (Issue #82: now optional)', async () => {
      // Arrange
      mockRequest = {
        params: {
          symbol: 'EURUSD',
          timeframe: 'H1',
        },
        query: {},
      };

      (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockResolvedValue([]);

      // Act
      await getMarketData(mockRequest as Request, mockResponse as Response, mockNext);

      // Assert
      // Issue #82: Parameters are now optional, should not throw error
      expect(mockNext).toHaveBeenCalledWith(); // Called with no error
      expect(marketDataQueries.getEarliestTimestamp).toHaveBeenCalledWith('EURUSD', 'H1');
      expect(marketDataQueries.getLatestTimestamp).toHaveBeenCalledWith('EURUSD', 'H1');
    });
  });

  describe('getMarketData - Error Handling', () => {
    it('should call next with ValidationError for invalid inputs', async () => {
      // Arrange
      mockRequest = {
        params: {
          symbol: 'EURUSD',
          timeframe: 'H1',
        },
        query: {
          start: '2024-01-01T00:00:00Z',
          end: '2024-01-31T23:59:59Z',
          limit: '1000',
        },
      };

      const validationError = new ValidationError('Invalid timeframe', {
        field: 'timeframe',
        value: 'H2',
      });

      (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockRejectedValue(
        validationError
      );

      // Act
      await getMarketData(mockRequest as Request, mockResponse as Response, mockNext);

      // Assert
      expect(mockNext).toHaveBeenCalledWith(validationError);
    });

    it('should call next with QueryError for database failures', async () => {
      // Arrange
      mockRequest = {
        params: {
          symbol: 'EURUSD',
          timeframe: 'H1',
        },
        query: {
          start: '2024-01-01T00:00:00Z',
          end: '2024-01-31T23:59:59Z',
          limit: '1000',
        },
      };

      const queryError = new QueryError('Database query failed', new Error('Connection timeout'));

      (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockRejectedValue(queryError);

      // Act
      await getMarketData(mockRequest as Request, mockResponse as Response, mockNext);

      // Assert
      expect(mockNext).toHaveBeenCalledWith(queryError);
    });

    it('should call next with unexpected errors', async () => {
      // Arrange
      mockRequest = {
        params: {
          symbol: 'EURUSD',
          timeframe: 'H1',
        },
        query: {
          start: '2024-01-01T00:00:00Z',
          end: '2024-01-31T23:59:59Z',
          limit: '1000',
        },
      };

      const unexpectedError = new Error('Unexpected error');

      (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockRejectedValue(
        unexpectedError
      );

      // Act
      await getMarketData(mockRequest as Request, mockResponse as Response, mockNext);

      // Assert
      expect(mockNext).toHaveBeenCalledWith(unexpectedError);
    });

    it('should handle invalid date strings', async () => {
      // Arrange
      mockRequest = {
        params: {
          symbol: 'EURUSD',
          timeframe: 'H1',
        },
        query: {
          start: 'invalid-date',
          end: '2024-01-31T23:59:59Z',
          limit: '1000',
        },
      };

      // Act
      await getMarketData(mockRequest as Request, mockResponse as Response, mockNext);

      // Assert
      expect(mockNext).toHaveBeenCalledWith(
        expect.objectContaining({
          message: expect.stringContaining('Invalid date'),
        })
      );
    });

    it('should handle invalid limit values', async () => {
      // Arrange
      mockRequest = {
        params: {
          symbol: 'EURUSD',
          timeframe: 'H1',
        },
        query: {
          start: '2024-01-01T00:00:00Z',
          end: '2024-01-31T23:59:59Z',
          limit: 'not-a-number',
        },
      };

      // Act
      await getMarketData(mockRequest as Request, mockResponse as Response, mockNext);

      // Assert
      expect(mockNext).toHaveBeenCalledWith(
        expect.objectContaining({
          message: expect.stringContaining('Invalid limit'),
        })
      );
    });
  });

  describe('getMarketData - Response Format', () => {
    it('should store data in res.locals for middleware', async () => {
      // Arrange
      mockRequest = {
        params: {
          symbol: 'EURUSD',
          timeframe: 'H1',
        },
        query: {
          start: '2024-01-01T00:00:00Z',
          end: '2024-01-31T23:59:59Z',
          limit: '100',
        },
      };

      const mockData = [
        {
          rateTime: new Date('2024-01-15T12:00:00Z'),
          open: 1.085,
          high: 1.0875,
          low: 1.084,
          close: 1.0865,
          volume: 1500,
          indicators: {},
        },
      ];

      (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockResolvedValue(mockData);

      // Act
      await getMarketData(mockRequest as Request, mockResponse as Response, mockNext);

      // Assert
      expect(mockResponse.locals?.['data']).toEqual(mockData);
      expect(mockNext).toHaveBeenCalledWith();
    });

    it('should store empty array when no data found', async () => {
      // Arrange
      mockRequest = {
        params: {
          symbol: 'EURUSD',
          timeframe: 'H1',
        },
        query: {
          start: '2024-01-01T00:00:00Z',
          end: '2024-01-31T23:59:59Z',
          limit: '1000',
        },
      };

      (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockResolvedValue([]);

      // Act
      await getMarketData(mockRequest as Request, mockResponse as Response, mockNext);

      // Assert
      expect(mockResponse.locals?.['data']).toEqual([]);
      expect(mockNext).toHaveBeenCalledWith();
    });
  });

  describe('getMarketData - Optional Parameters (Issue #82)', () => {
    describe('No parameters provided', () => {
      it('should accept request with no start/end parameters and fetch full dataset', async () => {
        // Arrange
        mockRequest = {
          params: {
            symbol: 'EURUSD',
            timeframe: 'H1',
          },
          query: {},
        };

        const mockData = [
          {
            rateTime: new Date('2024-01-15T12:00:00Z'),
            open: 1.085,
            high: 1.0875,
            low: 1.084,
            close: 1.0865,
            volume: 1500,
            indicators: {
              sma_20: 1.0855,
            },
          },
        ];

        // Mock the helper functions that will be called
        (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockResolvedValue(mockData);

        // Act
        await getMarketData(mockRequest as Request, mockResponse as Response, mockNext);

        // Assert
        expect(mockResponse.locals?.['data']).toEqual(mockData);
        expect(mockNext).toHaveBeenCalledWith();
        // When no params provided, should call with earliest/latest timestamps
        expect(marketDataQueries.getMarketDataWithIndicators).toHaveBeenCalledWith(
          'EURUSD',
          'H1',
          expect.any(Date), // earliest timestamp from DB
          expect.any(Date), // latest timestamp from DB
          1000 // default limit
        );
      });

      it('should use default limit when no parameters provided', async () => {
        // Arrange
        mockRequest = {
          params: {
            symbol: 'EURUSD',
            timeframe: 'H1',
          },
          query: {}, // No limit specified
        };

        (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockResolvedValue([]);

        // Act
        await getMarketData(mockRequest as Request, mockResponse as Response, mockNext);

        // Assert
        const call = (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mock.calls[0];
        expect(call[4]).toBe(1000); // Default limit
      });

      it('should handle empty dataset when no data exists', async () => {
        // Arrange
        mockRequest = {
          params: {
            symbol: 'EURUSD',
            timeframe: 'H1',
          },
          query: {},
        };

        (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockResolvedValue([]);

        // Act
        await getMarketData(mockRequest as Request, mockResponse as Response, mockNext);

        // Assert
        expect(mockResponse.locals?.['data']).toEqual([]);
        expect(mockNext).toHaveBeenCalledWith();
      });

      it('should return data with indicators when available', async () => {
        // Arrange
        mockRequest = {
          params: {
            symbol: 'EURUSD',
            timeframe: 'H1',
          },
          query: {},
        };

        const mockData = [
          {
            rateTime: new Date('2024-10-21T12:00:00Z'),
            open: 1.09,
            high: 1.0925,
            low: 1.089,
            close: 1.091,
            volume: 2000,
            indicators: {
              sma_20: 1.0895,
              ema_50: 1.0888,
              rsi_14: 62.5,
              macd_line: 0.0012,
              bb_upper_20: 1.0935,
              bb_lower_20: 1.0855,
            },
          },
        ];

        (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockResolvedValue(mockData);

        // Act
        await getMarketData(mockRequest as Request, mockResponse as Response, mockNext);

        // Assert
        expect(mockResponse.locals?.['data']).toEqual(mockData);
        expect(mockResponse.locals?.['data'][0].indicators).toBeDefined();
        expect(Object.keys(mockResponse.locals?.['data'][0].indicators).length).toBeGreaterThan(0);
      });
    });

    describe('Start parameter only', () => {
      it('should accept request with only start parameter', async () => {
        // Arrange
        mockRequest = {
          params: {
            symbol: 'EURUSD',
            timeframe: 'H1',
          },
          query: {
            start: '2024-01-01T00:00:00Z',
          },
        };

        (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockResolvedValue([]);

        // Act
        await getMarketData(mockRequest as Request, mockResponse as Response, mockNext);

        // Assert
        expect(mockNext).toHaveBeenCalledWith();
        expect(marketDataQueries.getMarketDataWithIndicators).toHaveBeenCalledWith(
          'EURUSD',
          'H1',
          new Date('2024-01-01T00:00:00Z'),
          expect.any(Date), // latest timestamp from DB
          1000
        );
      });

      it('should return data from start to latest timestamp', async () => {
        // Arrange
        mockRequest = {
          params: {
            symbol: 'EURUSD',
            timeframe: 'H1',
          },
          query: {
            start: '2024-06-01T00:00:00Z',
          },
        };

        const mockData = [
          {
            rateTime: new Date('2024-10-21T12:00:00Z'), // Latest data
            open: 1.09,
            high: 1.0925,
            low: 1.089,
            close: 1.091,
            volume: 2000,
            indicators: {},
          },
        ];

        (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockResolvedValue(mockData);

        // Act
        await getMarketData(mockRequest as Request, mockResponse as Response, mockNext);

        // Assert
        expect(mockResponse.locals?.['data']).toEqual(mockData);
        expect(mockNext).toHaveBeenCalledWith();
      });

      it('should validate start date when only start provided', async () => {
        // Arrange
        mockRequest = {
          params: {
            symbol: 'EURUSD',
            timeframe: 'H1',
          },
          query: {
            start: 'invalid-date-format',
          },
        };

        // Act
        await getMarketData(mockRequest as Request, mockResponse as Response, mockNext);

        // Assert
        expect(mockNext).toHaveBeenCalledWith(
          expect.objectContaining({
            message: expect.stringContaining('Invalid date'),
          })
        );
      });

      it('should handle start parameter with custom limit', async () => {
        // Arrange
        mockRequest = {
          params: {
            symbol: 'EURUSD',
            timeframe: 'H1',
          },
          query: {
            start: '2024-01-01T00:00:00Z',
            limit: '5000',
          },
        };

        (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockResolvedValue([]);

        // Act
        await getMarketData(mockRequest as Request, mockResponse as Response, mockNext);

        // Assert
        const call = (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mock.calls[0];
        expect(call[2]).toEqual(new Date('2024-01-01T00:00:00Z'));
        expect(call[4]).toBe(5000);
      });
    });

    describe('End parameter only', () => {
      it('should accept request with only end parameter', async () => {
        // Arrange
        mockRequest = {
          params: {
            symbol: 'EURUSD',
            timeframe: 'H1',
          },
          query: {
            end: '2024-12-31T23:59:59Z',
          },
        };

        (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockResolvedValue([]);

        // Act
        await getMarketData(mockRequest as Request, mockResponse as Response, mockNext);

        // Assert
        expect(mockNext).toHaveBeenCalledWith();
        expect(marketDataQueries.getMarketDataWithIndicators).toHaveBeenCalledWith(
          'EURUSD',
          'H1',
          expect.any(Date), // earliest timestamp from DB
          new Date('2024-12-31T23:59:59Z'),
          1000
        );
      });

      it('should return data from earliest to end timestamp', async () => {
        // Arrange
        mockRequest = {
          params: {
            symbol: 'EURUSD',
            timeframe: 'H1',
          },
          query: {
            end: '2024-06-30T23:59:59Z',
          },
        };

        const mockData = [
          {
            rateTime: new Date('2024-01-01T00:00:00Z'), // Earliest data
            open: 1.08,
            high: 1.0825,
            low: 1.079,
            close: 1.081,
            volume: 1800,
            indicators: {},
          },
        ];

        (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockResolvedValue(mockData);

        // Act
        await getMarketData(mockRequest as Request, mockResponse as Response, mockNext);

        // Assert
        expect(mockResponse.locals?.['data']).toEqual(mockData);
        expect(mockNext).toHaveBeenCalledWith();
      });

      it('should validate end date when only end provided', async () => {
        // Arrange
        mockRequest = {
          params: {
            symbol: 'EURUSD',
            timeframe: 'H1',
          },
          query: {
            end: 'not-a-valid-date',
          },
        };

        // Act
        await getMarketData(mockRequest as Request, mockResponse as Response, mockNext);

        // Assert
        expect(mockNext).toHaveBeenCalledWith(
          expect.objectContaining({
            message: expect.stringContaining('Invalid date'),
          })
        );
      });

      it('should handle end parameter with custom limit', async () => {
        // Arrange
        mockRequest = {
          params: {
            symbol: 'EURUSD',
            timeframe: 'H1',
          },
          query: {
            end: '2024-12-31T23:59:59Z',
            limit: '2500',
          },
        };

        (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockResolvedValue([]);

        // Act
        await getMarketData(mockRequest as Request, mockResponse as Response, mockNext);

        // Assert
        const call = (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mock.calls[0];
        expect(call[3]).toEqual(new Date('2024-12-31T23:59:59Z'));
        expect(call[4]).toBe(2500);
      });
    });

    describe('Both parameters (regression tests)', () => {
      it('should maintain backward compatibility with both start and end parameters', async () => {
        // Arrange
        mockRequest = {
          params: {
            symbol: 'EURUSD',
            timeframe: 'H1',
          },
          query: {
            start: '2024-01-01T00:00:00Z',
            end: '2024-01-31T23:59:59Z',
            limit: '500',
          },
        };

        (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockResolvedValue([]);

        // Act
        await getMarketData(mockRequest as Request, mockResponse as Response, mockNext);

        // Assert
        expect(mockNext).toHaveBeenCalledWith();
        expect(marketDataQueries.getMarketDataWithIndicators).toHaveBeenCalledWith(
          'EURUSD',
          'H1',
          new Date('2024-01-01T00:00:00Z'),
          new Date('2024-01-31T23:59:59Z'),
          500
        );
      });

      it('should work with both parameters and default limit', async () => {
        // Arrange
        mockRequest = {
          params: {
            symbol: 'EURUSD',
            timeframe: 'H1',
          },
          query: {
            start: '2024-01-01T00:00:00Z',
            end: '2024-01-31T23:59:59Z',
          },
        };

        (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockResolvedValue([]);

        // Act
        await getMarketData(mockRequest as Request, mockResponse as Response, mockNext);

        // Assert
        const call = (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mock.calls[0];
        expect(call[4]).toBe(1000); // Default limit
      });

      it('should return correct data range with both parameters', async () => {
        // Arrange
        mockRequest = {
          params: {
            symbol: 'EURUSD',
            timeframe: 'H1',
          },
          query: {
            start: '2024-01-01T00:00:00Z',
            end: '2024-01-31T23:59:59Z',
          },
        };

        const mockData = [
          {
            rateTime: new Date('2024-01-15T12:00:00Z'),
            open: 1.085,
            high: 1.0875,
            low: 1.084,
            close: 1.0865,
            volume: 1500,
            indicators: {
              sma_20: 1.0855,
            },
          },
        ];

        (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockResolvedValue(mockData);

        // Act
        await getMarketData(mockRequest as Request, mockResponse as Response, mockNext);

        // Assert
        expect(mockResponse.locals?.['data']).toEqual(mockData);
        expect(mockResponse.locals?.['data'][0].rateTime.getTime()).toBeGreaterThanOrEqual(
          new Date('2024-01-01T00:00:00Z').getTime()
        );
        expect(mockResponse.locals?.['data'][0].rateTime.getTime()).toBeLessThanOrEqual(
          new Date('2024-01-31T23:59:59Z').getTime()
        );
      });
    });

    describe('Edge cases and error handling', () => {
      it('should handle database errors gracefully', async () => {
        // Arrange
        mockRequest = {
          params: {
            symbol: 'EURUSD',
            timeframe: 'H1',
          },
          query: {},
        };

        const dbError = new QueryError(
          'Database connection failed',
          new Error('Connection timeout')
        );

        (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockRejectedValue(dbError);

        // Act
        await getMarketData(mockRequest as Request, mockResponse as Response, mockNext);

        // Assert
        expect(mockNext).toHaveBeenCalledWith(dbError);
      });

      it('should handle validation errors from database layer', async () => {
        // Arrange
        mockRequest = {
          params: {
            symbol: 'EURUSD',
            timeframe: 'H1',
          },
          query: {
            start: '2024-02-01T00:00:00Z',
            end: '2024-01-01T00:00:00Z', // End before start
          },
        };

        const validationError = new ValidationError(
          'End date must be greater than or equal to start date',
          {
            field: 'end',
            value: new Date('2024-01-01T00:00:00Z'),
            constraint: 'end >= start',
          }
        );

        (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockRejectedValue(
          validationError
        );

        // Act
        await getMarketData(mockRequest as Request, mockResponse as Response, mockNext);

        // Assert
        expect(mockNext).toHaveBeenCalledWith(validationError);
      });

      it('should handle future dates in start parameter', async () => {
        // Arrange
        mockRequest = {
          params: {
            symbol: 'EURUSD',
            timeframe: 'H1',
          },
          query: {
            start: '2030-01-01T00:00:00Z', // Future date
          },
        };

        const validationError = new ValidationError('Start date cannot be in the future', {
          field: 'start',
          value: new Date('2030-01-01T00:00:00Z'),
        });

        (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockRejectedValue(
          validationError
        );

        // Act
        await getMarketData(mockRequest as Request, mockResponse as Response, mockNext);

        // Assert
        expect(mockNext).toHaveBeenCalledWith(validationError);
      });

      it('should handle maximum limit boundary', async () => {
        // Arrange
        mockRequest = {
          params: {
            symbol: 'EURUSD',
            timeframe: 'H1',
          },
          query: {
            limit: '10000', // MAX_LIMIT
          },
        };

        (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockResolvedValue([]);

        // Act
        await getMarketData(mockRequest as Request, mockResponse as Response, mockNext);

        // Assert
        const call = (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mock.calls[0];
        expect(call[4]).toBe(10000);
        expect(mockNext).toHaveBeenCalledWith();
      });

      it('should handle limit exceeding maximum', async () => {
        // Arrange
        mockRequest = {
          params: {
            symbol: 'EURUSD',
            timeframe: 'H1',
          },
          query: {
            limit: '15000', // Over MAX_LIMIT
          },
        };

        const validationError = new ValidationError('Limit cannot exceed 10000', {
          field: 'limit',
          value: 15000,
          constraint: 'limit <= 10000',
        });

        (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockRejectedValue(
          validationError
        );

        // Act
        await getMarketData(mockRequest as Request, mockResponse as Response, mockNext);

        // Assert
        expect(mockNext).toHaveBeenCalledWith(validationError);
      });

      it('should handle zero or negative limit', async () => {
        // Arrange
        mockRequest = {
          params: {
            symbol: 'EURUSD',
            timeframe: 'H1',
          },
          query: {
            limit: '0',
          },
        };

        const validationError = new ValidationError('Limit must be greater than 0', {
          field: 'limit',
          value: 0,
          constraint: 'limit > 0',
        });

        (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockRejectedValue(
          validationError
        );

        // Act
        await getMarketData(mockRequest as Request, mockResponse as Response, mockNext);

        // Assert
        expect(mockNext).toHaveBeenCalledWith(validationError);
      });

      it('should handle different timeframes with optional parameters', async () => {
        // Arrange - Test with different timeframe
        mockRequest = {
          params: {
            symbol: 'EURUSD',
            timeframe: 'D1',
          },
          query: {},
        };

        (marketDataQueries.getMarketDataWithIndicators as jest.Mock).mockResolvedValue([]);

        // Act
        await getMarketData(mockRequest as Request, mockResponse as Response, mockNext);

        // Assert
        expect(marketDataQueries.getMarketDataWithIndicators).toHaveBeenCalledWith(
          'EURUSD',
          'D1',
          expect.any(Date),
          expect.any(Date),
          1000
        );
      });
    });
  });
});
